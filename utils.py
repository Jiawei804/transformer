import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from tensorflow import keras


# 位置编码
# PE(pos,2i) = sin(pos/10000^(2i/d_model))
# PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

# pos: 位置 pos.shape = (seq_len,1)
# i: 维度 i.shape = (1,d_model)
# d_model: 模型维度, d_model.shape = (1,1)
def get_angles(pos, i, d_model):
    angles = 1 / np.power(10000,
                          (2 * (i // 2)) / np.float32(d_model))
    return pos * angles


# 计算位置编码
def get_position_embedding(sentence_len, d_model):
    # sentence_len 和 d_model 都是标量，先变为矩阵
    # pos是0到39，词的位置，i从0到511，和d_model相等，是设置的超参数
    # 对于长度相等的句子位置编码是一样的
    # angle_rads.shape = (sentence_len, 512)
    angle_rads = get_angles(np.arange(sentence_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    # 拼接成位置编码
    position_embedding = np.concatenate([sines, cosines], axis=-1)
    # 进行维度扩展，为了符合后续输入的要求，[sentence_len, d_model] -> [1, sentence_len, d_model]
    position_embedding = position_embedding[np.newaxis, :]
    # 变为float32类型，模型输入需要
    return tf.cast(position_embedding, dtype=tf.float32)


# 设置掩码
# 1.padding_mask
def create_padding_mask(batch_data):
    """

    :param batch_data: shape (batch_size, sentence_len)
    :return:
    """
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), dtype=tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


# 2.decoder mask
# 第一个位置代表第一个单词和自己的attention，第二位置是第二个单词和第一个单词的attention
# 看不到后面的词刚好是下三角，使用库函数tf.linalg.band_part来实现
# 模型在预测时是串行的，不应该注意到未预测出的词的信息
# [[1, 0, 0],
#  [4, 5, 0],
#  [7, 8, 9]]
# 前面看不到后面的padding，矩阵下面全部为0
# 在mask里，应该被忽略的我们会设成1，应该被保留的会设成0，
# 而如果mask相应位置上为1，那么我们就给对应的logits
# 加上一个超级小的负数， -1000000000， 这样，
# 对应的logits也就变成了一个超级小的数。然后在计算softmax的时候，
# 一个超级小的数的指数会无限接近与0。也就是它对应的attention的权重就是0了,
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # 这里1-下三角矩阵，对角线元素也为0
    return mask


# 缩放点积注意力
# 也叫自注意力
# 公式：Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    seq_len_v 与 seq_len_k 相等
    depth_v 与 depth 可以相等也可以不等，看具体的设计
    :param q: shape = (..., seq_len_q, depth)
    :param k: shape = (..., seq_len_k, depth)
    :param v: shape = (..., seq_len_v, depth_v)
    :param mask: shape = (..., seq_len_q, seq_len_k)
    :return:
    -output: weighted sum
    - attention_weights: weights of attention
    """
    # 计算attention只用了最后两维
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 拿到depth，为了对结果缩放
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 如果设计掩码，就加mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # scaled_attention_logits.shape = (seq_len_q, seq_len_k)
    # attention_weights.shape = (seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# 多头注意力
class MultiHeadAttention(keras.layers.Layer):
    """
    理论上:
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0

    实战中:
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0

    实战中技巧: q乘以W得到一个大的Q，然后分割为多个小q，拿每一个小q去做缩放点积
    q -> Wq -> Q -> split -> q0, q1, q2...
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0

        # 多头注意力拆分成多个缩放点积注意力
        self.depth = self.d_model // self.num_heads
        # 构造多头注意力所需要的层
        # 神经元个数是512, 这里传入的d_model就是512
        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)
        # 合并多头注意力之后要接全连接层
        self.dense = keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x (64, 40, 512) -> (64, 40, 8, 64)  (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))

        # 轴滚动 (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]  # 形状的第0维就是batch_size
        # 经过Q K V变化
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 开始做缩放点积，注意力信息存在num_heads，depth上
        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 做轴交换，还原到原来的形状
        # 在transpose之前，scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_v, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])

        # 合并计算出的8个缩放点积注意力
        # scaled_attention_outputs.shape: (batch_size, seq_len_v, num_heads, depth)
        # concat_attention.shape: (batch_size, seq_len_v, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))

        # 最后经过一个全连接层输出
        output = self.dense(concat_attention)
        return output, attention_weights


# 前馈神经网络
# 先升维，再降维
def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model),
    ])


# 学习率变化，是先增后减，因为前期可以快点，后期模型比较好，就要慢点 warm-up
# lrate = (d_model ** -0.5) * min(step_num ** (-0.5),
#                                 step_num * warm_up_steps **(-1.5))
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# 图像展示encoder和decoder融合时的attention
def plot_encoder_decoder_attention(attention, input_sentence,
                                   result, layer_name,
                                   pt_tokenizer, en_tokenizer):
    fig = plt.figure(figsize=(16, 8))

    input_id_sentence = pt_tokenizer.encode(input_sentence)
    print(len(input_id_sentence), len(result))
    # attention.shape: (num_heads, tar_len, input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        ax.matshow(attention[head][:-1, :])

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)
        ax.set_yticklabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size],
            fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


# 评估模型，使用bleu指标，微软的nltk框架
def bleu_score(val_examples, pt_tokenizer, en_tokenizer, evaluate):
    total_bleu = 0.0
    for pt_val, en_val in val_examples:
        if len(pt_tokenizer.encode(pt_val.numpy())) > 38 or len(en_tokenizer.encode(en_val.numpy())) > 38:
            continue
        result, _ = evaluate(pt_val.numpy())
        predicted_sentence = en_tokenizer.decode([i for i in result
                                                  if i < en_tokenizer.vocab_size])
        bleu = sentence_bleu(
            [en_val.numpy().decode('utf8').split(' ')],
            predicted_sentence.split(' '),
            weights=(0.25, 0.25, 0.25, 0.25))  # weights是四个指标的权重，即1-gram,2-gram,3-gram,4-gram
        total_bleu += bleu

    return total_bleu / len(val_examples)


if __name__ == '__main__':
    from settings import *


    # 测试：热力图展示位置编码
    def plot_position_embedding(position_embedding):
        plt.pcolormesh(position_embedding[0, ::-1, :], cmap='RdBu')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()


    position_embedding = get_position_embedding(sentence_len=50, d_model=512)
    plot_position_embedding(position_embedding)


    # 测试：查看学习率变化曲线
    def plot_learning_rate_schedule(temp_learning_rate_schedule):
        plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        plt.show()


    temp_learning_rate_schedule = CustomSchedule(D_MODEL)
    plot_learning_rate_schedule(temp_learning_rate_schedule)

    # 测试：查看掩码
    # 设置3x5矩阵，0都是padding，是零的得到的都是1，其他的都是零
    x = tf.constant([[7, 6, 0, 0, 0], [1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    print(create_padding_mask(x))
    print(create_look_ahead_mask(5))


    # 测试：缩放点积注意力
    # 测试
    def print_scaled_dot_product_attention(q, k, v):
        temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
        print("Attention weights are:")
        print(temp_att)
        print("Output is:")
        print(temp_out)


    # 定义一个测试的Q，K，V
    temp_q1 = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)
    print_scaled_dot_product_attention(temp_q1, temp_k, temp_v)

    # 测试：多头注意力
    x = tf.random.uniform((1, 40, 256))  # (batch_size, seq_len_q, dim)
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    # 开始计算，把y既当q，又当k，v
    output, attn = temp_mha(x, x, x, mask=None)
    print(f"MultiHeadAttention output: {output.shape}")  # 输出的尺寸，和x的尺寸一致
    print(f"MultiHeadAttention weights: {attn.shape}")  # 注意力的尺寸

    # 测试：前馈神经网络
    sample_ffn = feed_forward_network(512, 2048)
    # 给一个输入测试
    print(f"ffn output: {sample_ffn(tf.random.uniform((64, 50, 512))).shape}")
