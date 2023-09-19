
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.layers import GlobalAvgPool2D, Flatten, Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.losses import mse


class customized_renet_model:
    def __init__(self, input_shape=(512, 512, 3), loss=mse, weights='imagenet',
                 learning_rate=0.001, dropout_rate=0.2):
        self.input_shape = input_shape
        self.loss = loss
        self.weights = weights
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def create(self):
        self.base_model = ResNet50(input_shape=self.input_shape, include_top=False, weights=self.weights)
        x = self.base_model.output
        x = GlobalAvgPool2D()(x)
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dense(units=10, activation='relu')(x)
        #x = Dropout(self.dropout_rate)(x)
        x = Dense(units=1, activation='linear')(x)
        #x = Dense(units=1, activation='sigmoid')(x)
        self.model = Model(inputs=self.base_model.input, outputs=x)

    def compile(self):
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)


# def customized_renet_model(input_shape=(512, 512, 3), weights='imagenet'):
#     base_model = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
#     x = base_model.output
#     x = GlobalAvgPool2D()(x)
#     x = Flatten()(x)
#     x = Dense(units=512, activation='linear')(x)
#     x = LeakyReLU()(x)
#     x = Dense(units=10, activation='linear')(x)
#     x = LeakyReLU()(x)
#     x = Dense(units=1, activation='linear')(x)
#     x = LeakyReLU()(x)
#     model = Model(inputs=base_model.input, outputs=x)
#     return model
