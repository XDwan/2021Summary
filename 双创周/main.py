    def _build_model(self):

        model = Sequential()
        model.add(
            layers.Conv2D(
                filters=6,kernel_size=(5,5),
                strides=(1,1),input_shape=(28,28,1),
                padding = 'valid',activation='relu',
                kernel_initializer='uniform'
            )
        )
        model.add(
            layers.MaxPooling2D(pool_size=(2,2))
        )
        model.add(
            layers.Conv2D(
                64,(5,5),strides=(1,1),padding='valid',
                activation='relu',kernel_initializer='uniform'
            )
        )
        model.add(
            layers.MaxPooling2D(
                pool_size=(2,2)
            )
        )
        model.add(
            layers.Flatten()
        )
        model.add(
            layers.Dense(256,activation='relu')
        )
        model.add(
            layers.Dense(128,activation='relu')
        )

        model.add(
            layers.Dense(10,activation='softmax')
        )

        model.summary()

        return model