from tensorflow import keras
from models import EncoderModel, DecoderModel


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_length, rate):
        super().__init__()
        self.encoder = EncoderModel(
            num_layers, input_vocab_size, max_length,
                                    d_model, num_heads, dff, rate)

        self.decoder = DecoderModel(
            num_layers, target_vocab_size, max_length,
                                    d_model, num_heads, dff, rate)

        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        # inp.shape: (batch_size, inp_seq_len)
        # encoding_output.shape: (batch_size, inp_seq_len, d_model)
        encoding_output = self.encoder(
            inp, training, encoder_padding_mask)

        # decoding_output.shape: (batch_size, tar_seq_len, d_model)
        decoding_output, attention_weights = self.decoder(
            tar, encoding_output, training,
            decoder_mask, encoder_decoder_padding_mask)

        # prediction.shape: (batch_size, tar_seq_len, target_vocab_size)
        prediction = self.final_layer(decoding_output)

        return prediction, attention_weights


if __name__=='__main__':
    import tensorflow as tf
    # 测试
    sample_transformer = Transformer(4, 128, 8, 512,
                                     8216, 8089, 40,
                                       rate=0.1)
    temp_input = tf.random.uniform((64, 26))
    temp_target = tf.random.uniform((64, 31))

    # 得到输出
    predictions, attention_weights = sample_transformer(
        temp_input, temp_target, training=False,
        encoder_padding_mask=None,
        decoder_mask=None,
        encoder_decoder_padding_mask=None)
    # 输出shape
    print(predictions.shape)
    print('-' * 50)
    # attention_weights 的shape打印，为了后面画图做铺垫
    for key in attention_weights:
        print(key, attention_weights[key].shape)
    print('-' * 50)
    sample_transformer.summary()

