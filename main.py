import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from settings import *
from transformer import Transformer
from utils import create_padding_mask, create_look_ahead_mask, CustomSchedule, plot_encoder_decoder_attention, \
    bleu_score

# 葡萄牙语Portugal -> 英语English，基于subword_level
# examples: <class 'dict'>, 有train和validation两个key
examples, info = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

# 制作语料库，subword_level
# 制作西班牙语的tokenizer
pt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                                                        [pt.numpy() for pt, en in train_examples],
                                                        target_vocab_size=2**13)

# 制作英语的tokenizer
en_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                                                        [en.numpy() for pt, en in train_examples],
                                                        target_vocab_size=2**13)

# 词表大小加2
INPUT_VOCAB_SIZE = pt_tokenizer.vocab_size + 2
TARGET_VOCAB_SIZE = en_tokenizer.vocab_size + 2
print(f"INPUT_VOCAB_SIZE: {INPUT_VOCAB_SIZE}, TARGET_VOCAB_SIZE: {TARGET_VOCAB_SIZE}")

# 将两种语言的文本转化为subword形式，[pt_tokenizer.vocab_size] 和 [pt_tokenizer.cocab_size+1]
# 分别表示一句话的开始标记和结束标记，<start>,<end>
def encode_to_subword(pt_sentence, en_sentence):
    pt_sentence = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(pt_sentence.numpy()) + [pt_tokenizer.vocab_size + 1]
    en_sentence = [en_tokenizer.vocab_size] + en_tokenizer.encode(en_sentence.numpy()) + [en_tokenizer.vocab_size + 1]
    return pt_sentence, en_sentence

# 去掉较长的样本，只要葡萄牙语和英语同时小于40(tokens的长度)的样本（只是为了计算能力较弱的本地可以运行）
def filter_by_max_length(pt, en):
    return tf.math.logical_and(tf.size(pt) <= MAX_LENGTH,
                            tf.size(en) <= MAX_LENGTH)

# 用py_function把python函数包装成tensorflow，加快执行效率
def tf_encode_to_subword(pt_sentence, en_sentence):
    return tf.py_function(encode_to_subword,
                          [pt_sentence, en_sentence],
                          [tf.int64, tf.int64])

train_dataset = train_examples.map(tf_encode_to_subword)
train_dataset = train_dataset.filter(filter_by_max_length)
# 洗牌，分batch，在inp 与 out上分别取每个batch最大长度做为填充标准
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                                                padded_shapes=([-1],[-1])) # 在最后一个维度上做填充

# 验证集只做了过滤和padding，没有洗牌
valid_dataset = val_examples.map(tf_encode_to_subword)
valid_dataset = valid_dataset.filter(filter_by_max_length).padded_batch(BATCH_SIZE,
                                                                        padded_shapes=([-1],[-1]))

# 初始化mask矩阵
def create_masks(inp, tar):
    """
    Encoder:
     - encoder_padding_mask (self attention of EncoderLayer)
     对于encoder，padding值没有意义，无需attention
     Decoder:
     - decoder_padding_mask (self attention of DecoderLayer)
     decoder也有padding，所以mask掉
     - look_ahead_mask (self attention of DecoderLayer)
     target位置上的词不能看到后面的词，因为后面的词还没有预测出来
     - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
     decoder不应该到encoder的padding上花费精力
    :param inp:
    :param tar:
    :return:
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = encoder_padding_mask

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask,
                              look_ahead_mask)

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask



transformer = Transformer(
    NUM_LAYERS, # encoder和decoder的层数
    INPUT_VOCAB_SIZE, # 输入词表的大小
    TARGET_VOCAB_SIZE, # 输出词表的大小
    MAX_LENGTH, # 最大长度
    D_MODEL, # 模型的维度
    NUM_HEADS, # 多头注意力的头数
    DFF, # feed forward 层的神经元数
    DROPOUT_RATE # dropout比例
)

learning_rate = CustomSchedule(D_MODEL)
# 优化器每次会调用learning_rate，把step传给它，得到学习率
optimizer = keras.optimizers.Adam(learning_rate,
                                  beta_1 = 0.9,
                                  beta_2 = 0.98,
                                  epsilon = 1e-9)

# 设置检查点，如果训练过程中断不至于从头训练
checkpoint_dir = './py_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                        transformer=transformer)


train_loss = keras.metrics.Mean(name = 'train_loss') # 累积求batch平均损失
train_accuracy = keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy') # 累积求预测准确率


loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
# 计算损失
def loss_function(real, pred):
    # 损失做了掩码处理，是padding的地方不计算损失
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)



# 原始的标签是这样的 <start>  I  hava a bag <end>

# Decoder input   <start>  I  hava a bag
# Decoder output  I  hava a bag <end>

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]   # 没有end decoder模块输入
    tar_real = tar[:, 1:]   # 没有start decoder模块输出

    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,
                                     encoder_padding_mask,
                                     decoder_mask,
                                     encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables)
    )
    train_loss(loss) # 损失，累积效果
    train_accuracy(tar_real, predictions) # 准确率，累积效果


# 如果加载成功，是<tensorflow.python.checkpoint.checkpoint.CheckpointLoadStatus，不成功有init
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print(status)

for epoch in range(EPOCHS):
    start = time.time()
    # reset后就会从零开始累积
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):

        train_step(inp, tar)

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch+1, batch, train_loss.result(),
                train_accuracy.result()
            ))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time take for 1 epoch: {} secs\n'.format(
        time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)

# loss是一个正常的指标，而accuracy只是机器翻译的一个参考指标，可以看趋势，业界并不以准确率作为模型好坏的参考指标

"""
eg: ABCDEFGH
Train: ABCDEFG
Eval:   A -> B
        AB -> C
        ABC -> D
transformer模型的训练过程是这样的，每次训练的时候，输入是ABCDE，输出是BCDEF，这样就可以训练出A->B,AB->C,ABC->D,ABCD->E,ABCDE->F
transformer可以并行训练，因为它的每一层都是独立的，可以并行计算，这样就可以加快训练速度
"""
# 翻译函数，输入葡萄牙语，输出英语
def evaluate(inp_sentence):
    # 输入的句子需要经过编码
    input_id_sentence = ([pt_tokenizer.vocab_size] + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size + 1])
    # encoder的输入是一个batch，所以需要给它增加一个维度
    # encoder_input.shape: (1, input_sentence_length)
    encoder_input = tf.expand_dims(input_id_sentence, 0)

    # decoder的第一个输入是start，shape为(1,1)
    # 预测一个词就放入decoder_input， decoder_input给多个就可以预测多个，我们给一个
    decoder_input = tf.expand_dims([en_tokenizer.vocab_size], 0) # start token，相当于<start>传进去

    for i in range(MAX_LENGTH):
        # 产生mask并传给transformer
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(encoder_input, decoder_input)
        # predictions.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(
            encoder_input,
            decoder_input,
            False,
            encoder_padding_mask,
            decoder_mask,
            encoder_decoder_padding_mask)
        # prediction.shape: (batch_size, target_vocab_size)
        # 每次预测最后一个词，所以取最后一个词
        # print(predictions.shape)
        prediction = predictions[:, -1, :] # 取最后一个词
        # 预测值是对每个词的概率分布，取概率最大的那个词
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        # 如果predicted_id是end token，就返回结果
        if tf.equal(predicted_id, en_tokenizer.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0), attention_weights
        # 把预测的词和decoder_input拼起来，作为下一次的输入
        # decoder_input.shape: (1 , seq_len)
        # print(decoder_input.shape, predicted_id.shape)
        decoder_input = tf.concat([decoder_input, [predicted_id]], axis=-1)

    return tf.squeeze(decoder_input, axis=0), attention_weights


# 翻译 如果layer_name不为空，就会画出attention图
def translate(input_sentence, layer_name=''):
    result, attention_weights = evaluate(input_sentence)

    predicted_sentence = en_tokenizer.decode(
        [i for i in result if i < en_tokenizer.vocab_size])
    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))
    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence,
                                       result, layer_name, pt_tokenizer, en_tokenizer)





if __name__ == '__main__':
    translate(
        'isto é minha vida')
    # 展示翻译效果，并绘制热力图
    # translate(
    #     'isto é minha vida',  # This is my life.
    #     layer_name='decoder_layer4_att2')

    # bleu = bleu_score(val_examples, pt_tokenizer, en_tokenizer, evaluate)
    # print('bleu score: {:.4f}'.format(bleu))