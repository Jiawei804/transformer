from tensorflow import keras
import tensorflow as tf

from utils import MultiHeadAttention, feed_forward_network, get_position_embedding


class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward ->  add & normalize & dropout
    """


    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """

        :param d_model: self attention 和 feed_forward_network
        :param num_heads: self_attention
        :param dff: feed_forward_network
        :param rate: dropout
        """
        super().__init__()
        # d_model 是多头注意力中WQ, WK, WV的神经元个数
        # 多头注意力和前馈神经网络
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        # epsilon 平滑项，防止除0问题发生
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6)

        # dropout，默认以10%的几率被dropout掉
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)


    def call(self, x, training, encoder_padding_mask):
        # x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
        # encoder模块里，x作为q，k，v
        attn_output, _ = self.mha(x, x, x, encoder_padding_mask)
        attn_output = self.dropout1(attn_output, training = training)
        # 残差连接，dim 必须等于 d_model，才可以相加
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.norm2(out1 + ffn_output)

        return out2


class DecoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalization & dropout -> out1
    out1, encoder_output -> attention -> add & normalization & dropout -> out2
    out2 -> ffn -> add & normalization & dropout -> out3
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.norm3 = keras.layers.LayerNormalization(
            epsilon=1e-6)

    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.norm1(x + attn1)

        attn2, attn_weights2 = self.mha2(
            out1, encoding_outputs, encoding_outputs,
                encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.norm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm3(ffn_output + out2)

        return out3, attn_weights1, attn_weights2


# encoder模型
class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(input_vocab_size, self.d_model)

        self.position_embedding = get_position_embedding(max_length, self.d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)
        ]

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            input_seq_len, self.max_length,
            "input_seq_len should be less or equal to self.max_length"
        )

        x = self.embedding(x)
        # x.shape: (batch_size, input_seq_len, d_model)
        # x做缩放，是值在0到d_model之间
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # 因为x长度比position_embedding可能要小，因此embedding切片后和x相加
        # position_embedding的轴0的size是1，在x的轴零维度广播相加
        x += self.position_embedding[:, :input_seq_len, :]

        x = self.dropout(x, training=training)

        # x输入到下一层
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        # x最终shape如下
        # x.shape: (batch_size, input_seq_len, d_model)
        return x


# decoder模型
class DecoderModel(keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(target_vocab_size, self.d_model)

        self.position_embedding = get_position_embedding(max_length, self.d_model)

        self.dropout = keras.layers.Dropout(rate)
        self.decoder_layers = [
            DecoderLayer(d_model,num_heads,dff,rate)
            for _ in range(self.num_layers)
        ]

    def call(self, x, encoding_outputs, training,
             decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        # 如果要输出的都超出了max_length，就报错
        tf.debugging.assert_less_equal(
            output_seq_len, self.max_length,
            "input_seq_len should be less or equal to self.max_length"
        )

        # attention_weights都是由decoder layer返回，把它保存下来
        attention_weights = {}

        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)

        # 根据d_model进行缩放
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 把x加上位置编码
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            # attn1,attn2分别是两个attention
            x, attn1, attn2 = self.decoder_layers[i](
                x, encoding_outputs, training,
                decoder_mask, encoder_decoder_padding_mask
            )
            attention_weights[
                'decoder_layer{}_att1'.format(i+1)]= attn1
            attention_weights[
                'decoder_layer{}_att2'.format(i + 1)] = attn2

            # x.shape: (batch_size, output_seq_len, d_model)
            # attention_weights是为了画图

        return x, attention_weights


if __name__ == '__main__':
    # encoder测试
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_input = tf.random.uniform((64, 50, 512))  # 因为做了残差连接，所以x的形状必须和mha输出形状一致
    sample_output = sample_encoder_layer(sample_input, False, None)
    print(f"encoder_layer output: {sample_output.shape}")

    sample_encoder_model = EncoderModel(2, 8500, 40,
                                        512, 8, 2048)
    sample_encoder_model_input = tf.random.uniform((64, 37))
    sample_encoder_model_output = sample_encoder_model(
        sample_encoder_model_input, False, encoder_padding_mask=None)
    print(f"encoder_model output: {sample_encoder_model_output.shape}")

    print('-' * 50)
    # decoder测试
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_input = tf.random.uniform((64, 60, 512))
    sample_decoder_output, sample_decoder_attn_weights1, sample_decoder_attn_weights2 = sample_decoder_layer(
        sample_decoder_input, sample_output, False, None, None)
    print(f"decoder_layer output: {sample_decoder_output.shape}")
    print(sample_decoder_output.shape)
    print(f"decoder_layer_attn_weights1 output: {sample_decoder_attn_weights1.shape}")  # 最后一维60是和x的维度一致的
    print(f"decoder_layer_attn_weights2 output: {sample_decoder_attn_weights2.shape}")  # 最后一维60是和x的维度相关的

    sample_decoder_model = DecoderModel(2, 8000, 40,
                                        512, 8, 2048)
    sample_decoder_model_input = tf.random.uniform((64, 35))
    sample_decoder_model_output, sample_decoder_model_att = sample_decoder_model(
        sample_decoder_model_input,
        sample_encoder_model_output,  # 注意这里是encoder的output
        training=False, decoder_mask=None,
        encoder_decoder_padding_mask=None)
    print(f"decoder_model output: {sample_decoder_model_output.shape}")
