# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/9/5 08:44
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : models.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V3.4: 2024/8/21 - 2024/8/30
#             1. 增加额外的消融实验：去掉残差连接和InceptionTime模块(即仅保留跨时间和通道注意力模块)；
#             2. 增加多变量时间序列中交互方式的对比：将跨时间和通道注意力替换为传统的LSTM（即改变序列间的交互方式）
#             V3.3: 2023/10/13 - 2023/10/18
#             1. 大论文、添加注意力权重可视化以实现可解释性
#             V3.2: 2023/3/30
#             1. 各种模型的消融与比较，添加公开数据集
#             V2.0: 2023/2/1 - 2023/2/4, 2023/2/12
#             1. 添加结构化自发言语任务实验，并和非结构化对比；
#             2. 增加针对模型的消融实验；
#             V1.0 - ZL.Z：2022/9/5 - 2022/9/13
#             First version.
# @License  : None
# @Brief    : 模型

from config import *
import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, \
    Activation, MultiHeadAttention, Conv1D, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, \
    Permute, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from attention import Attention
from typing import Union, Tuple
import pathlib as pl
import time
import warnings
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import scikit_posthocs as skp
import re
from keras_bert import AdamWarmup, calc_train_steps
import gc

warnings.filterwarnings("ignore", message="Custom mask layers require a config.*")
warnings.filterwarnings("ignore", message="The `lr` argument is deprecated.*")


class MixPooling1D(Layer):
    def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
        super(MixPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        # Initialize MaxPooling1D and AveragePooling1D layers
        self.max_pool = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)
        self.avg_pool = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)
        # Initialize weights for mixing max and avg pools
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs, *args, **kwargs):
        # Apply MaxPooling1D and AveragePooling1D
        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)
        # Mix the results
        mixed_pool = self.alpha * max_pool + (1 - self.alpha) * avg_pool
        return mixed_pool

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        steps = input_shape[1]
        features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size,
                                               self.padding,
                                               self.strides)
        return tensor_shape.TensorShape([input_shape[0], length, features])

    def get_config(self):
        config = super(MixPooling1D, self).get_config()
        config.update({'pool_size': self.pool_size, 'strides': self.strides, 'padding': self.padding})
        return config


class InceptionTimeNetwork:
    def __init__(self, n_classes: int = 2, n_filters: int = 32, strides: int = 1, padding='same',
                 activation='linear', use_bottleneck: bool = True, use_residual: bool = True,
                 kernel_size: int = 41, depth: int = 6, model_name: str = None, input_shape: Tuple[int] = None,
                 output_directory: Union[str, pl.Path] = None, build: bool = False, verbose: bool = False,):
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = 32
        self.kernel_size = kernel_size - 1
        self.use_residual = use_residual
        self.depth = depth
        self.callbacks = None
        self.y_categories = None
        if isinstance(model_name, str) and not model_name == "":
            self.model_name = model_name
        else:
            self.model_name = "InceptionTimeNetwork"
        if build:
            self.output_directory = output_directory
            self.input_shape = input_shape
            self.model = self.build_model()
            if verbose:
                self.model.summary()
            self.verbose = verbose
            file_name = self.model_name + "_init.h5"
            self.model.save_weights(self.output_directory.joinpath(file_name))

    def inception_module(self, input_tensor):
        # the inception module is a bottleneck operation followed by 3 parallel convolutions and
        # a maximum pooling operation followed by a convolution with kernel size 1
        # only apply bottleneck operation if the input is multivariante data!
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                input_shape=input_tensor.shape[1:],
                padding=self.padding,
                activation=self.activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor
        # create list of kernel sizes of the convolutions (100%, 50%, 25% of the input kernel_size)
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]
        # create a list of multiple, distinct convolutions on same input (that is the output of input_inception)
        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(
                Conv1D(
                    filters=self.n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=self.strides,
                    padding=self.padding,
                    activation=self.activation,
                    use_bias=False,
                )(input_inception)
            )
        # parallel path: add maximum pooling to same input (that is the output of input_inception)
        max_pool_1 = MaxPooling1D(pool_size=3, strides=self.strides, padding=self.padding)(input_tensor)
        # convolve the output of max-pooling with kernel size 1 (this is basically a scaling)
        conv_6 = Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=False,
        )(max_pool_1)
        # append to list of operations
        conv_list.append(conv_6)
        # create inception module: concatenate all operations that they run in parallel and add batch normalization for
        # better training (vanishing gradient problem)
        inception_module = Concatenate(axis=2)(conv_list)
        inception_module = BatchNormalization()(inception_module)
        # set activation functions to ReLU
        inception_module = Activation(activation="relu")(inception_module)
        return inception_module

    def shortcut_layer(self, input_tensor, out_tensor):
        # 1D convolution followed by batch normalization in parallel to "normal" input-output
        n_filters = int(out_tensor.shape[-1])
        shortcut_y = Conv1D(filters=n_filters, kernel_size=1, strides=self.strides,
                            padding=self.padding, use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)
        # put shortcut in parallel to the "normal" layer
        block = Add()([shortcut_y, out_tensor])
        block = Activation("relu")(block)
        return block

    def build_model(self) -> keras.Model:
        # define shape of the expected input
        input_layer = Input(shape=self.input_shape)
        # initialize first layer as the input layer
        x = input_layer
        # initialize short-cut layer with input layer
        input_res = input_layer
        # stack inceltion modules / blocks
        for d in range(self.depth):
            x = self.inception_module(x)
            # x = InceptionTimeModule(self.n_filters, self.strides, self.padding, self.activation, self.use_bottleneck,
            #                         self.use_residual, self.kernel_size)(x)
            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer(input_res, x)
                input_res = x
        # penultimate layer is a global average pooling layer
        gap_layer = GlobalAveragePooling1D()(x)
        # output layer is a dense softmax layer for classification
        output_layer = Dense(self.n_classes, activation="softmax")(gap_layer)
        # stack all layers together to a model
        model = Model(inputs=input_layer, outputs=output_layer)
        # compile setting loss function and optimizer
        model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        # construct / set callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
        # add checkpoints
        file_name = self.model_name + "_best_model.h5"
        file_path = self.output_directory.joinpath(file_name)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor="loss", save_best_only=True)
        # set callbacks
        self.callbacks = [reduce_lr, model_checkpoint]
        return model

    def fit(
        self,
        x_train: Union[np.ndarray, pd.Series, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_val: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None, batch_size: int = 64, n_epochs: int = 1500,
    ) -> keras.Model:
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # convert label input (y) to categoricals and store for backtransformation
        y_train = pd.Categorical(y_train)
        self.y_categories = y_train.categories
        # extract category codes because keras only handles increasing integers as classes starting at 0
        y_train = y_train.codes
        y_val = pd.Categorical(y_val, categories=self.y_categories).codes
        if batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = batch_size
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_data = None
        hist = self.model.fit(
            x_train, y_train,
            batch_size=mini_batch_size,
            epochs=n_epochs,
            verbose=self.verbose,
            validation_data=validation_data,
            callbacks=self.callbacks,
        )
        # add label categories to output file
        file_name = self.model_name + "_best_model.h5"
        self.model.save(self.output_directory.joinpath(file_name))
        return self.model

    def predict(self, x: Union[np.ndarray, pd.Series, pd.DataFrame], batch_size: int = 64) -> np.ndarray:
        if self.y_categories is None:
            raise ValueError("No label categories found. Perhaps the model was not trained yet?")
        y_pred = self.model.predict(x, batch_size=batch_size).argmax(axis=1)
        return y_pred


class BaselineDMFCCModel:
    """基线模型：基于EMD的改进MFCC特征（DMFCC）的SVM模型"""

    def __init__(self, data_file: str = "", model_name: str = "DMFCC-CEEMDAN", task_name: str = "PD"):
        """
        初始化
        :param data_file: 数据集文件
        :param model_name: 模型名称
        :param task_name: 使用的言语任务：结构化（PD）和非结构化（SI）
        """
        data_file = os.path.normpath(data_file)
        data_name = os.path.basename(data_file).split('_')[-1].rstrip('.pkl')
        if data_file.endswith('pkl'):
            _feat_data = pd.read_pickle(data_file)  # type: pd.DataFrame
            if task_name in ["PD", "SI"]:
                nan_rows = _feat_data[_feat_data['dmfcc'].isna()]  # 排除由于模态数不足，模态分解失败的样本，并保证不同任务的样本一致
                nan_ids = nan_rows['id'].values
                nan_augs = nan_rows['aug'].values
                feat_data = _feat_data[~((_feat_data['id'].isin(nan_ids)) & (_feat_data['aug'].isin(nan_augs)))]
            else:
                feat_data = _feat_data.copy()
            feat_data = feat_data[feat_data["task"] == task_name].reset_index(drop=True)
        else:
            raise ValueError('无效数据，仅接受.pkl数据集文件')
        y_label = np.array(feat_data['label'])
        _x_data_clf = feat_data["dmfcc"]
        x_data_clf = []
        for i_subj in _x_data_clf:
            x_data_clf.append(i_subj)
        x_data_clf = np.array(x_data_clf, dtype=np.float32)  # shape=[样本数，特征维数]=[1782,156]
        # print(x_data_clf.shape)
        train_data_clf, self.train_label = x_data_clf, y_label
        ss = StandardScaler()  # 标准化特征
        pipe = Pipeline([('ss', ss)])
        self.train_data_clf = pipe.fit_transform(train_data_clf)
        self.model_clf_save_dir = f'./models/{data_name}/{task_name}/{model_name}'  # 保存模型路径
        if not os.path.exists(self.model_clf_save_dir):
            os.makedirs(self.model_clf_save_dir)
        self.model_name = model_name
        self.task_name = task_name
        self.n_folders = 5
        self.subj = np.array(feat_data['id'])

    def model_train_evaluate(self, fit: bool = True):
        """
        模型训练
        :param fit: 是否进行训练
        :return: 分类模型
        """
        res = {'acc_cv': [], 'f1_cv': [], 'roc_auc_cv': [], 'sen_cv': [], 'spec_cv': [], 'training_time': []}
        skf = StratifiedGroupKFold(n_splits=self.n_folders, shuffle=True, random_state=rs)  # n折交叉验证，分层采样
        fold_index = 0
        for train_index, test_index in skf.split(self.train_data_clf, self.train_label, self.subj):
            fold_index += 1
            print(f'------- FOLD {fold_index} / {skf.get_n_splits(groups=self.subj)} -------')
            save_model = os.path.join(self.model_clf_save_dir, f'model_{self.model_name}_{fold_index}.h5')
            if fit:
                model_svc = SVC(C=10, gamma='auto', probability=True, random_state=rs)
                # 使用CalibratedClassifierCV概率校准估计器SVC，避免后期predict和predict_proba结果不一致
                model_clf = CalibratedClassifierCV(model_svc, n_jobs=1)  # 仅校准最优参数对应的模型
                start_t = time.time()
                model_clf.fit(self.train_data_clf[train_index], self.train_label[train_index])
                res['training_time'].append(time.time() - start_t)
                joblib.dump(model_clf, save_model)  # 保存模型
            else:
                if os.path.exists(save_model):  # 存在已训练模型且设置加载，
                    print("----------加载分类模型：{}----------".format(save_model))
                    model_clf = joblib.load(save_model)  # 加载已训练模型
                    res['training_time'].append(np.NAN)
                else:
                    raise FileNotFoundError("分类模型不存在，无法评估，请先训练")
            y_preds = model_clf.predict(self.train_data_clf[test_index])
            y_pred_proba = model_clf.predict_proba(self.train_data_clf[test_index])
            acc = accuracy_score(self.train_label[test_index], y_preds)
            roc_auc = roc_auc_score(self.train_label[test_index], y_pred_proba[:, 1])
            precision, recall, f1_score, support = precision_recall_fscore_support(self.train_label[test_index], y_preds,
                                                                                   average='binary')  # 测试集各项指标
            sen = recall
            spec = scorer_sensitivity_specificity(self.train_label[test_index], y_preds)
            res['acc_cv'].append(acc * 100)
            res['f1_cv'].append(f1_score * 100)
            res['roc_auc_cv'].append(roc_auc * 100)
            res['sen_cv'].append(sen * 100)
            res['spec_cv'].append(spec * 100)
            # print(acc, f1_score, sen, spec, roc_auc)
        # 输出最优参数的n折交叉验证的各项指标
        print(f"CV Accuracy: {np.mean(res['acc_cv']):.2f}±{np.std(res['acc_cv']):.2f}")
        print(f"CV F1 score: {np.mean(res['f1_cv']):.2f}±{np.std(res['f1_cv']):.2f}")
        print(f"CV Sensitivity (Recall): {np.mean(res['sen_cv']):.2f}±{np.std(res['sen_cv']):.2f}")
        print(f"CV Specificity: {np.mean(res['spec_cv']):.2f}±{np.std(res['spec_cv']):.2f}")
        print(f"CV ROC-AUC: {np.mean(res['roc_auc_cv']):.2f}±{np.std(res['roc_auc_cv']):.2f}")
        print(f"CV Training Time/s: {np.mean(res['training_time']):.2f}±{np.std(res['training_time']):.2f}")
        return res


class BaselineEmbeddingModel:
    """基线模型：基于Embedding（w2v2+hubert）的BiLSTM+Attention模型"""

    def __init__(self, data_file: str = "", model_name: str = "BiLSTM-Attention", task_name='PD'):
        """
        初始化
        :param data_file: 数据集文件
        :param model_name: 模型名称
        :param task_name: 使用的自发言语任务：结构化（PD）和非结构化（SI）
        """
        data_file = os.path.normpath(data_file)
        data_name = os.path.basename(data_file).split('_')[-1].rstrip('.pkl')
        if data_file.endswith('pkl'):
            feat_data = pd.read_pickle(data_file)  # type: pd.DataFrame
            feat_data = feat_data[feat_data["task"] == task_name].reset_index(drop=True)
        else:
            raise ValueError('无效数据，仅接受.pkl数据集文件')
        y_label = np.array(feat_data['label'])
        x_data_clf = []
        for i_subj in feat_data.index:
            x_data_clf.append(np.hstack((feat_data["w2v2"][i_subj], feat_data["hubert"][i_subj])))
        x_data_clf = np.array(x_data_clf, dtype=np.float32)  # shape=[样本数，时间序列帧数，特征维数]=[1800,499,1536]
        # print(x_data_clf.shape)
        self.train_data_clf, self.train_label = x_data_clf, y_label
        self.model_clf_save_dir = f'./models/{data_name}/{task_name}/{model_name}'  # 保存模型路径
        if not os.path.exists(self.model_clf_save_dir):
            os.makedirs(self.model_clf_save_dir)
        self.model_name = model_name
        self.task_name = task_name
        self.data_name = data_name
        self.n_folders = 5
        self.batch_size = 16
        self.epochs = 30
        self.subj = np.array(feat_data['id'])

    def model_create(self):
        """
        使用子类Keras Sequential API方式构建模型
        :return: 返回模型
        """
        model = Sequential()
        model.add(Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1])))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(Attention(32))
        model.add(Dense(16))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(1, activation="sigmoid"))
        # 编译模型：损失函数采用分类交叉熵，优化采用Adam，将识别准确率作为模型评估
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_train_evaluate(self, fit: bool = True):
        """
        模型训练
        :param fit: 是否进行训练
        :return: 交叉验证结果
        """
        res = {'acc_cv': [], 'f1_cv': [], 'roc_auc_cv': [], 'sen_cv': [], 'spec_cv': [], 'training_time': []}
        skf = StratifiedGroupKFold(n_splits=self.n_folders, shuffle=True, random_state=rs)  # n折交叉验证，分层采样
        es_callback = EarlyStopping(monitor='loss', patience=10)
        fold_index = 0
        for train_index, test_index in skf.split(self.train_data_clf, self.train_label, self.subj):
            fold_index += 1
            print(f'------- FOLD {fold_index} / {skf.get_n_splits(groups=self.subj)} -------')
            save_model = os.path.join(self.model_clf_save_dir, f'model_{self.model_name}_{fold_index}.h5')
            log_dir = os.path.join(self.model_clf_save_dir, f'logs/model_{self.model_name}_{fold_index}')
            if fit:
                model = self.model_create()
                # 使用回调函数保存.h5类型模型，保存权重、网络结构、参数等训练最好的信息,每个epoch保存一次
                cp_callback = ModelCheckpoint(filepath=save_model, save_best_only=True, monitor="loss")
                callbacks = [es_callback, cp_callback, TensorBoard(log_dir)]
                start_t = time.time()
                model.fit(self.train_data_clf[train_index], self.train_label[train_index],
                          batch_size=self.batch_size, epochs=self.epochs, verbose=0,
                          # validation_data=(self.train_data_clf[test_index], self.train_label[test_index]),
                          shuffle=True, callbacks=callbacks)
                res['training_time'].append(time.time() - start_t)
            else:
                if os.path.exists(save_model):  # 存在已训练模型且设置加载，
                    print("----------加载分类模型：{}----------".format(save_model))
                    res['training_time'].append(np.NAN)
                else:
                    raise FileNotFoundError("分类模型不存在，无法评估，请先训练")
            model_best = load_model(save_model, custom_objects={'Attention': Attention, 'MixPooling1D': MixPooling1D,
                                                                'AdamWarmup': AdamWarmup})
            try:  # 标签为1的概率
                y_pred_proba = model_best(self.train_data_clf[test_index]).numpy()[:, 1]  # sparse_categorical_crossentropy
            except IndexError:
                y_pred_proba = model_best(self.train_data_clf[test_index]).numpy().flatten()  # binary_crossentropy
            y_pred_label = (y_pred_proba > 0.5).astype("int32")
            acc = accuracy_score(self.train_label[test_index], y_pred_label)
            roc_auc = roc_auc_score(self.train_label[test_index], y_pred_proba)
            precision, recall, f1_score, support = \
                precision_recall_fscore_support(self.train_label[test_index], y_pred_label,
                                                average='binary', zero_division=1)  # 测试集各项指标
            sen = recall
            spec = scorer_sensitivity_specificity(self.train_label[test_index], y_pred_label)
            res['acc_cv'].append(acc * 100)
            res['f1_cv'].append(f1_score * 100)
            res['roc_auc_cv'].append(roc_auc * 100)
            res['sen_cv'].append(sen * 100)
            res['spec_cv'].append(spec * 100)
            # print(acc, f1_score, sen, spec, roc_auc)
            if fit:
                del model
                del model_best
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                setup_seed(rs)
                gc.collect()
        # 输出最优参数的n折交叉验证的各项指标
        print(f"CV Accuracy: {np.mean(res['acc_cv']):.2f}±{np.std(res['acc_cv']):.2f}")
        print(f"CV F1 score: {np.mean(res['f1_cv']):.2f}±{np.std(res['f1_cv']):.2f}")
        print(f"CV Sensitivity (Recall): {np.mean(res['sen_cv']):.2f}±{np.std(res['sen_cv']):.2f}")
        print(f"CV Specificity: {np.mean(res['spec_cv']):.2f}±{np.std(res['spec_cv']):.2f}")
        print(f"CV ROC-AUC: {np.mean(res['roc_auc_cv']):.2f}±{np.std(res['roc_auc_cv']):.2f}")
        print(f"CV Training Time/s: {np.mean(res['training_time']):.2f}±{np.std(res['training_time']):.2f}")
        return res


class CrossTCAInceptionTimeModel(BaselineEmbeddingModel):
    """基于Embedding（w2v2+hubert）的跨时间、跨通道的多头注意力机制InceptionTime模型"""

    def __init__(self, num_heads: int = 4, depth: int = 5, **kwargs):
        """
        初始化
        :param num_heads: 多头注意力机制的头数，默认为4
        :param depth: InceptionTime模块层数，即深度，默认为5
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        if self.model_name == "w2v-CTCAIT":
            self.train_data_clf = self.train_data_clf[:, :, :self.train_data_clf.shape[-1] // 2]
        elif self.model_name == "HuBERT-CTCAIT":
            self.train_data_clf = self.train_data_clf[:, :, self.train_data_clf.shape[-1] // 2:]
        self.inception = InceptionTimeNetwork(depth=depth)
        self.n_folders = 5
        self.batch_size = 16
        self.epochs = 30

    def model_create(self):
        """
        构建模型
        :return: 返回模型
        """
        if self.model_name == "CTCAIT_dropT":
            model = self.model_create_dropT()
        elif self.model_name == "CTCAIT_dropC":
            model = self.model_create_dropC()
        elif self.model_name == "CTCAIT_dropTC":
            model = self.model_create_dropTC()
        elif self.model_name == "CTCAIT_dropR":
            model = self.model_create_dropR()
        elif self.model_name == "IT":
            model = self.model_create_IT()
        elif self.model_name == "CTCA":
            model = self.model_create_CTCA()
        else:
            model = self.model_create_all()
        return model

    def model_create_all(self):
        """
        构建模型
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        # Cross-variable attention
        inputs_c = Permute((2, 1))(inputs_downsample)
        layer_mha_c = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        # mha_output_c=[batch,chan,seq]=(None, 512, 499)
        # mha_scores_c=[batch,head,chan,chan]=(None, 4, 512, 512)
        mha_output_c, mha_scores_c = layer_mha_c(query=inputs_c, value=inputs_c,
                                                 return_attention_scores=True)
        mha_output_c = Permute((2, 1))(mha_output_c)  # channels_last,shape=(None, 499, 512)
        # print(mha_output_c, '\n', mha_scores_c)
        # stack inception modules
        inc_output = inputs_downsample  # initialize first layer as the input layer
        input_res = inputs_downsample  # initialize short-cut layer with input layer
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        # Cross-temporal attention
        layer_mha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        # mha_output_t=[batch,seq,chan]=(None, 499, 128)
        # mha_scores_t=[batch,head,seq,seq]=(None, 4, 499, 499)
        mha_output_t, mha_scores_t = layer_mha_t(query=block, value=block,
                                                 return_attention_scores=True)
        # print(mha_output_t, '\n', mha_scores_t)
        mha_output = Concatenate(axis=-1)([mha_output_t, mha_output_c])  # (None, 499, 128+512)
        layer_gap = GlobalAveragePooling1D()(mha_output)  # shape=(None, 128+512)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        # 编译模型：损失函数采用分类交叉熵，优化采用Adam，将识别准确率作为模型评估
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # model.summary()
        return model

    def model_create_dropT(self):
        """
        构建模型：去掉跨时间多头注意力
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inputs_c = Permute((2, 1))(inputs_downsample)
        layer_mha_c = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_c, mha_scores_c = layer_mha_c(query=inputs_c, value=inputs_c,
                                                 return_attention_scores=True)
        mha_output_c = Permute((2, 1))(mha_output_c)  # channels_last,shape=(None, 499, 512)
        inc_output = inputs_downsample
        input_res = inputs_downsample
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        mha_output = Concatenate(axis=-1)([block, mha_output_c])  # (None, 499, 128+512)
        layer_gap = GlobalAveragePooling1D()(mha_output)  # shape=(None, 128+512)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_dropC(self):
        """
        构建模型：去掉跨通道多头注意力
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inc_output = inputs_downsample
        input_res = inputs_downsample
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        layer_mha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_t, mha_scores_t = layer_mha_t(query=block, value=block,
                                                 return_attention_scores=True)
        layer_gap = GlobalAveragePooling1D()(mha_output_t)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_dropTC(self):
        """
        构建模型：去掉跨时间和跨通道多头注意力
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inc_output = inputs_downsample
        input_res = inputs_downsample
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        layer_gap = GlobalAveragePooling1D()(block)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_dropR(self):
        """
        构建模型：去掉残差连接
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inputs_c = Permute((2, 1))(inputs_downsample)
        layer_mha_c = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_c, mha_scores_c = layer_mha_c(query=inputs_c, value=inputs_c,
                                                 return_attention_scores=True)
        mha_output_c = Permute((2, 1))(mha_output_c)  # channels_last,shape=(None, 499, 512)
        inc_output = inputs_downsample
        input_res = inputs_downsample
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        layer_mha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_t, mha_scores_t = layer_mha_t(query=inc_output, value=inc_output,
                                                 return_attention_scores=True)
        mha_output = Concatenate(axis=-1)([mha_output_t, mha_output_c])  # (None, 499, 128+512)
        layer_gap = GlobalAveragePooling1D()(mha_output)  # shape=(None, 128+512)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_IT(self):
        """
        构建模型：仅保留InceptionTime模块
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inc_output = inputs_downsample
        input_res = inputs_downsample
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        layer_gap = GlobalAveragePooling1D()(inc_output)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_CTCA(self):
        """
        构建模型：去掉残差连接和InceptionTime模块，即仅保留交互：跨时间和通道注意力模块
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        inputs_c = Permute((2, 1))(inputs_downsample)
        layer_mha_c = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_c, mha_scores_c = layer_mha_c(query=inputs_c, value=inputs_c,
                                                 return_attention_scores=True)
        mha_output_c = Permute((2, 1))(mha_output_c)  # channels_last,shape=(None, 499, 512)
        layer_mha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_heads*2, dropout=0.05)
        mha_output_t, mha_scores_t = layer_mha_t(query=inputs_downsample, value=inputs_downsample,
                                                 return_attention_scores=True)
        mha_output = Concatenate(axis=-1)([mha_output_t, mha_output_c])  # (None, 499, 128+512)
        layer_gap = GlobalAveragePooling1D()(mha_output)  # shape=(None, 128+512)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


class LstmInteraInceptionTimeModel(CrossTCAInceptionTimeModel):
    """基于Embedding（w2v2+hubert）的跨时间、跨通道的LSTM交互InceptionTime模型"""

    def __init__(self, **kwargs):
        """
        初始化
        """
        super().__init__(**kwargs)
        if self.model_name == "CT_Att":
            self.model_name = "CTCAIT_dropC"
        elif self.model_name == "CC_Att":
            self.model_name = "CTCAIT_dropT"
        self.model_clf_save_dir = f'./models/{self.data_name}/{self.task_name}/{self.model_name}'

    def model_create(self):
        """
        构建模型
        :return: 返回模型
        """
        if self.model_name == "CTCAIT_dropTC":
            model = self.model_create_dropTC()
        elif self.model_name == "CT_LSTM":
            model = self.model_create_CTLSTM()
        elif self.model_name == "CC_LSTM":
            model = self.model_create_CCLSTM()
        elif self.model_name == "CTC_LSTM":
            model = self.model_create_CTCLSTM()
        elif self.model_name == "CTCAIT_dropC":
            model = self.model_create_dropC()
        elif self.model_name == "CTCAIT_dropT":
            model = self.model_create_dropT()
        else:
            model = self.model_create_all()
        return model

    def model_create_CTLSTM(self):
        """
        构建模型：LSTM作为跨时间交互模块
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        # stack inception modules
        inc_output = inputs_downsample  # initialize first layer as the input layer
        input_res = inputs_downsample  # initialize short-cut layer with input layer
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        # Cross-temporal LSTM
        lstm_t = LSTM(128, return_sequences=True)(block)  # shape=(None, 499, 128)
        layer_gap = GlobalAveragePooling1D()(lstm_t)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        # 编译模型：损失函数采用分类交叉熵，优化采用Adam，将识别准确率作为模型评估
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_CCLSTM(self):
        """
        构建模型：LSTM作为跨通道交互模块
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        # Cross-variable LSTM
        inputs_c = Permute((2, 1))(inputs_downsample)
        lstm_c = LSTM(128, return_sequences=True)(inputs_c)  # shape=(None, 512, 128)
        # stack inception modules
        inc_output = inputs_downsample  # initialize first layer as the input layer
        input_res = inputs_downsample  # initialize short-cut layer with input layer
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        cross_output = Concatenate(axis=1)([block, lstm_c])  # (None, 499+512, 128)
        layer_gap = GlobalAveragePooling1D()(cross_output)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        # 编译模型：损失函数采用分类交叉熵，优化采用Adam，将识别准确率作为模型评估
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def model_create_CTCLSTM(self):
        """
        构建模型：LSTM作为跨时间和跨通道交互模块
        :return: 返回模型
        """
        inputs = Input(shape=(self.train_data_clf.shape[1], self.train_data_clf.shape[-1]))
        inputs_downsample = Permute((2, 1))(MixPooling1D(pool_size=3)(Permute((2, 1))(inputs)))  # shape=(None, 499, 512)
        # Cross-variable LSTM
        inputs_c = Permute((2, 1))(inputs_downsample)
        lstm_c = LSTM(128, return_sequences=True)(inputs_c)  # shape=(None, 512, 128)
        # stack inception modules
        inc_output = inputs_downsample  # initialize first layer as the input layer
        input_res = inputs_downsample  # initialize short-cut layer with input layer
        for d in range(self.inception.depth):
            inc_output = self.inception.inception_module(inc_output)
            if self.inception.use_residual and d % 3 == 2:
                inc_output = self.inception.shortcut_layer(input_res, inc_output)
                input_res = inc_output
        block = self.inception.shortcut_layer(inputs_downsample, inc_output)  # shape=(None, 499, 128)
        # Cross-temporal LSTM
        lstm_t = LSTM(128, return_sequences=True)(block)  # shape=(None, 499, 128)
        cross_output = Concatenate(axis=1)([lstm_t, lstm_c])  # (None, 499+512, 128)
        layer_gap = GlobalAveragePooling1D()(cross_output)  # shape=(None, 128)
        layer_fc = Dense(32)(layer_gap)
        layer_bn = BatchNormalization()(layer_fc)
        layer_ac = Activation("swish")(layer_bn)
        outputs = Dense(2, activation="softmax")(layer_ac)
        model = Model(inputs=inputs, outputs=outputs)
        # 编译模型：损失函数采用分类交叉熵，优化采用Adam，将识别准确率作为模型评估
        n_samp = self.train_data_clf.shape[0] * (1 - 1 / self.n_folders) / self.batch_size
        total_steps, warmup_steps = calc_train_steps(num_example=n_samp, batch_size=self.batch_size,
                                                     epochs=self.epochs, warmup_proportion=0.2)
        opt = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # model.summary()
        return model


def scorer_sensitivity_specificity(y_true, y_pred, sen_spec=False):
    """
    敏感性特异性指标
    :param y_true: 真实值
    :param y_pred: 预测概率
    :param sen_spec: True返回特异性，否则返回敏感性
    :return: 敏感性、特异性
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    if sen_spec:
        return sen
    else:
        return spec


def model_compare(data_file: str = "", task_list=None, model_dict: dict = None,
                  perf_comp_f: str = "", mode: str = 'w', fit: bool = True, load_data=True) -> pd.DataFrame:
    """
    模型比较
    :param data_file: 数据集文件
    :param task_list: 使用的言语任务：结构化（PD）和非结构化（SI）
    :param model_dict: 模型名、模型类字典
    :param perf_comp_f: 模型性能比较的csv文件
    :param mode: csv保存文件模式
    :param fit: 是否进行训练
    :param load_data: 是否加载之前模型已评估好的结果直接获取指标数据
    :return: 各模型各配置的交叉验证结果
    """
    if task_list is None:
        task_list = ["PD", "SI"]
    if load_data:
        df_res = pd.read_csv(perf_comp_f)
    else:
        task_l, conf_l, acc_l, f1_l, sen_l, spe_l, auc_l, trt_l = [], [], [], [], [], [], [], []
        for tn in task_list:
            for ml in model_dict:
                print(f"------- Running {ml} of {tn} task -------\n")
                _model = model_dict[ml](data_file=data_file, model_name=ml, task_name=tn)
                res = _model.model_train_evaluate(fit=fit)
                task_l.append(tn)
                conf_l.append(ml)
                acc_l.append(f"{np.mean(res['acc_cv']):.2f} ({np.std(res['acc_cv']):.2f})")
                f1_l.append(f"{np.mean(res['f1_cv']):.2f} ({np.std(res['f1_cv']):.2f})")
                sen_l.append(f"{np.mean(res['sen_cv']):.2f} ({np.std(res['sen_cv']):.2f})")
                spe_l.append(f"{np.mean(res['spec_cv']):.2f} ({np.std(res['spec_cv']):.2f})")
                auc_l.append(f"{np.mean(res['roc_auc_cv']):.2f} ({np.std(res['roc_auc_cv']):.2f})")
                trt_l.append(f"{np.mean(res['training_time']):.2f} ({np.std(res['training_time']):.2f})")
        df_res = pd.DataFrame({"Task": task_l, "Feature-Model Config": conf_l, "Acc/%": acc_l, "F1/%": f1_l,
                               "Sen/%": sen_l, "Spe/%": spe_l, "AUC/%": auc_l, "Training Time/s": trt_l})
        with open(perf_comp_f, mode=mode) as f:  # 文件不存在时创建文件，文件已经存在时跳过列标题
            df_res.to_csv(f, header=f.tell() == 0, index=False)
    print(df_res)
    return df_res


def mha_head_influ(data_file: str = "", task_list=None, perf_comp_f: str = "",
                   mode: str = 'w', fit: bool = True, load_data=True) -> pd.DataFrame:
    """
    多头注意力机制的头数对模型的影响
    :param data_file: 数据集文件
    :param task_list: 使用的言语任务：结构化（PD）和非结构化（SI）
    :param perf_comp_f: 模型性能比较的csv文件
    :param mode: csv保存文件模式
    :param fit: 是否进行训练
    :param load_data: 是否加载之前模型已评估好的结果直接获取指标数据
    :return: 不同头数的多头注意力机制的交叉验证结果
    """
    if task_list is None:
        task_list = ["PD", "SI"]
    if load_data:
        df_res = pd.read_csv(perf_comp_f)
    else:
        task_l, conf_l, acc_l, f1_l, sen_l, spe_l, auc_l, trt_l = [], [], [], [], [], [], [], []
        for tn in task_list:
            for n_head in range(1, 9):
                print(f"------- Running CrossTCAInceptionTimeModel with {n_head} heads of {tn} task -------\n")
                _model = CrossTCAInceptionTimeModel(data_file=data_file, model_name=f"CTCAIT_h{n_head}",
                                                    task_name=tn, num_heads=n_head)
                res = _model.model_train_evaluate(fit=fit)
                task_l.append(tn)
                conf_l.append(f"CTCAIT_h{n_head}")
                acc_l.append(f"{np.mean(res['acc_cv']):.2f} ({np.std(res['acc_cv']):.2f})")
                f1_l.append(f"{np.mean(res['f1_cv']):.2f} ({np.std(res['f1_cv']):.2f})")
                sen_l.append(f"{np.mean(res['sen_cv']):.2f} ({np.std(res['sen_cv']):.2f})")
                spe_l.append(f"{np.mean(res['spec_cv']):.2f} ({np.std(res['spec_cv']):.2f})")
                auc_l.append(f"{np.mean(res['roc_auc_cv']):.2f} ({np.std(res['roc_auc_cv']):.2f})")
                trt_l.append(f"{np.mean(res['training_time']):.2f} ({np.std(res['training_time']):.2f})")
        df_res = pd.DataFrame({"Task": task_l, "Feature-Model Config": conf_l, "Acc/%": acc_l, "F1/%": f1_l,
                               "Sen/%": sen_l, "Spe/%": spe_l, "AUC/%": auc_l, "Training Time/s": trt_l})
        with open(perf_comp_f, mode=mode) as f:  # 文件不存在时创建文件，文件已经存在时跳过列标题
            df_res.to_csv(f, header=f.tell() == 0, index=False)
    print(df_res)
    delimiter = re.escape(' (')
    for tn in task_list:
        plt.figure(figsize=(9, 6), tight_layout=True)
        plt.xlabel('Number of Heads', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Metrics', fontdict={'family': font_family, 'size': 18})
        x_coords = range(1, df_res[df_res['Task'] == tn]['Task'].count() + 1)
        acc_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['Acc/%'].str.split(delimiter).tolist()]
        plt.plot(x_coords, acc_l, 'rs-', lw=1.5, ms=10, label='Acc')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['Acc/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords, y=acc_l, yerr=yerr, fmt='rs-', ms=10, capsize=2, label='Acc')
        f1_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['F1/%'].str.split(delimiter).tolist()]
        plt.plot(range(1, df_res[df_res['Task'] == tn]['Task'].count() + 1), f1_l, 'g^--', lw=1.5, ms=10, label='F1')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['F1/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords, y=f1_l, yerr=yerr, fmt='g^--', lw=1.5, ms=10, capsize=2, label='F1')
        auc_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['AUC/%'].str.split(delimiter).tolist()]
        plt.plot(range(1, df_res[df_res['Task'] == tn]['Task'].count() + 1), auc_l, 'b*-', lw=1.5, ms=10, label='AUC')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['AUC/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords, y=auc_l, yerr=yerr, fmt='b*-', lw=1.5, ms=10, capsize=2, label='AUC')
        plt.legend(loc="lower right", prop={'family': font_family, 'size': 18}, labelspacing=1.0, frameon=False)
        plt.ylim(0.75, 1.0)
        plt.xticks(fontsize=16, fontproperties=font_family)
        plt.yticks(fontsize=16, fontproperties=font_family)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = os.path.join(os.path.dirname(perf_comp_f), f'mhaHeadInflu/mhaHeadInflu_{tn}.png')
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    transparent=True)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.show()
        plt.close('all')
    return df_res


def depth_influ(data_file: str = "", task_list=None, perf_comp_f: str = "",
                mode: str = 'w', fit: bool = True, load_data=True) -> pd.DataFrame:
    """
    InceptionTime模块层数，即深度对模型的影响
    :param data_file: 数据集文件
    :param task_list: 使用的言语任务：结构化（PD）和非结构化（SI）
    :param perf_comp_f: 模型性能比较的csv文件
    :param mode: csv保存文件模式
    :param fit: 是否进行训练
    :param load_data: 是否加载之前模型已评估好的结果直接获取指标数据
    :return: 不同InceptionTime模块层数的交叉验证结果
    """
    if task_list is None:
        task_list = ["PD", "SI"]
    if load_data:
        df_res = pd.read_csv(perf_comp_f)
    else:
        task_l, conf_l, acc_l, f1_l, sen_l, spe_l, auc_l, trt_l = [], [], [], [], [], [], [], []
        for tn in task_list:
            for n_depth in range(1, 9):
                print(f"------- Running CrossTCAInceptionTimeModel with {n_depth} depth of {tn} task -------\n")
                _model = CrossTCAInceptionTimeModel(data_file=data_file, model_name=f"CTCAIT_d{n_depth}",
                                                    task_name=tn, depth=n_depth)
                res = _model.model_train_evaluate(fit=fit)
                task_l.append(tn)
                conf_l.append(f"CTCAIT_d{n_depth}")
                acc_l.append(f"{np.mean(res['acc_cv']):.2f} ({np.std(res['acc_cv']):.2f})")
                f1_l.append(f"{np.mean(res['f1_cv']):.2f} ({np.std(res['f1_cv']):.2f})")
                sen_l.append(f"{np.mean(res['sen_cv']):.2f} ({np.std(res['sen_cv']):.2f})")
                spe_l.append(f"{np.mean(res['spec_cv']):.2f} ({np.std(res['spec_cv']):.2f})")
                auc_l.append(f"{np.mean(res['roc_auc_cv']):.2f} ({np.std(res['roc_auc_cv']):.2f})")
                trt_l.append(f"{np.mean(res['training_time']):.2f} ({np.std(res['training_time']):.2f})")
        df_res = pd.DataFrame({"Task": task_l, "Feature-Model Config": conf_l, "Acc/%": acc_l, "F1/%": f1_l,
                               "Sen/%": sen_l, "Spe/%": spe_l, "AUC/%": auc_l, "Training Time/s": trt_l})
        with open(perf_comp_f, mode=mode) as f:  # 文件不存在时创建文件，文件已经存在时跳过列标题
            df_res.to_csv(f, header=f.tell() == 0, index=False)
    print(df_res)
    delimiter = re.escape(' (')
    for tn in task_list:
        plt.figure(figsize=(9, 6), tight_layout=True)
        plt.xlabel('Number of Layers', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Metrics', fontdict={'family': font_family, 'size': 18})
        wd = 0.2
        x_coords = np.arange(1, df_res[df_res['Task'] == tn]['Task'].count() + 1)
        acc_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['Acc/%'].str.split(delimiter).tolist()]
        plt.bar(x_coords - wd, acc_l, width=wd, color='r', label='Acc')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['Acc/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords - wd, y=acc_l, yerr=yerr, c='k', fmt='none', capsize=2)
        f1_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['F1/%'].str.split(delimiter).tolist()]
        plt.bar(x_coords, f1_l, width=wd, color='g', label='F1')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['F1/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords, y=f1_l, yerr=yerr, c='k', fmt='none', capsize=2)
        auc_l = [float(i[0])/100 for i in df_res[df_res['Task'] == tn]['AUC/%'].str.split(delimiter).tolist()]
        plt.bar(x_coords + wd, auc_l, width=wd, color='b', label='AUC')
        # yerr = [float(i[-1][:-1])/100 for i in df_res[df_res['Task'] == tn]['AUC/%'].str.split(delimiter).tolist()]
        # plt.errorbar(x=x_coords + wd, y=auc_l, yerr=yerr, c='k', fmt='none', capsize=2)
        plt.legend(loc="upper left", ncol=3, prop={'family': font_family, 'size': 18},
                   handlelength=1.5, handletextpad=0.3, columnspacing=1.0, frameon=False)
        plt.ylim(0.75, 1.02)
        plt.xticks(fontsize=16, fontproperties=font_family)
        plt.yticks(fontsize=16, fontproperties=font_family)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = os.path.join(os.path.dirname(perf_comp_f), f'depthInflu/depthInflu_{tn}.png')
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    transparent=True)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.show()
        plt.close('all')
    return df_res


def cd_diagram(md_com_data: pd.DataFrame, fig_save_dir: str = '', datasets_col: str = 'Task',
               models_col: str = 'Feature-Model Config', metrics_col: str = 'Acc/%') -> [pd.Series, pd.DataFrame]:
    """
    绘制显著性图，即p值热图，以及绘制多模型多数据集统计比较CD图，即Critical Difference Diagrams
    Friedman test with Nemenyi post-hoc test
    :param md_com_data: pd.DataFrame类型的模型性能比较数据
    :param fig_save_dir: 结果图片保存路径
    :param datasets_col: md_com_data中的不同数据集/情况的列名
    :param models_col: md_com_data中的不同模型/方法的列名
    :param metrics_col: md_com_data中的需要统计比较差异的指标列名
    :return: avg_rank：不同模型在不同数据集上的性能表现平均排名, test_results：带有事后检验的Friedman测试结果
    """
    rank_data = md_com_data.copy()
    rank_data['rank'] = rank_data.groupby(datasets_col)[metrics_col].rank()
    avg_rank = rank_data.groupby(models_col).mean()['rank']
    test_results = skp.posthoc_nemenyi_friedman(md_com_data, y_col=metrics_col, block_col=datasets_col,
                                                group_col=models_col, melted=True)
    # print(test_results)
    # 绘制显著性图，即p值热图
    plt.figure(figsize=(9, 7))
    # Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
    cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'square': True, 'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}
    ax, cbar = skp.sign_plot(test_results, **heatmap_args)
    rect = Rectangle((0, 0), test_results.shape[0], test_results.shape[-1], linewidth=3,
                     edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.tick_params(bottom=False, left=False)
    ax.set_xticklabels(ax.get_xticklabels(), fontdict={'family': font_family, 'size': 18},
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), fontdict={'family': font_family, 'size': 18}, rotation=0)
    cbar.set_ticks(list(np.linspace(0, 3, 4)), labels=['$p < 0.001$', '$p < 0.01$', '$p < 0.05$', 'ns'])
    cbar.ax.tick_params(labelsize=16)
    for sp in plt.gca().spines:
        plt.gca().spines[sp].set_color('k')
        plt.gca().spines[sp].set_linewidth(1)
    plt.subplots_adjust(left=0.17, bottom=0.13, right=0.85, top=0.97)
    fig_file = os.path.join(fig_save_dir, f'cdDiagram/sign_plot.png')
    if not os.path.exists(os.path.dirname(fig_file)):
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                transparent=True)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close('all')
    # 绘制CD图
    plt.figure(figsize=(10, 2), tight_layout=True)
    skp.critical_difference_diagram(avg_rank, test_results, text_h_margin=0.3,
                                    label_props={'fontsize': 18, 'fontproperties': font_family},
                                    crossbar_props={'linewidth': 2.5, 'marker': 'o'},
                                    label_fmt_left='{label}  ', label_fmt_right='  {label}',
                                    marker_props={'marker': '*', 's': 150, 'color': 'y', 'edgecolor': 'k'})
    plt.xticks(fontsize=14, fontproperties=font_family)
    plt.gca().spines['top'].set_color('k')
    plt.gca().spines['top'].set_linewidth(1.25)
    fig_file = os.path.join(fig_save_dir, f'cdDiagram/cdDiagram.png')
    if not os.path.exists(os.path.dirname(fig_file)):
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                transparent=True)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close('all')
    return avg_rank, test_results


def gradcam_vis(fig_save_dir: str = ''):
    """
    模型内部Grad-CAM及跨时间和通道注意力权重加权均值叠加在语音波形上的可视化，以进行模型预测解释
    :param fig_save_dir: 结果图片保存路径
    :return: None
    """
    import librosa
    import glob
    from scipy.interpolate import interp1d
    from mpl_toolkits.axisartist.parasite_axes import HostAxes
    import matplotlib.ticker as mtick
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch
    from sklearn.preprocessing import minmax_scale
    from keract import get_activations
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore
    from matplotlib.collections import LineCollection

    def func_format1(x, pos):
        return f"{x:.1f}"

    self = CrossTCAInceptionTimeModel(data_file=feat_f_wdhc, model_name='CTCAIT')
    save_model = os.path.join(self.model_clf_save_dir, f'model_{self.model_name}_1.h5')
    model = load_model(save_model, custom_objects={'Attention': Attention, 'MixPooling1D': MixPooling1D,
                                                   'AdamWarmup': AdamWarmup})
    # model.summary()
    feat_data_all = pd.read_pickle(feat_f_wdhc)
    feat_data_all = feat_data_all[feat_data_all['task'] == 'PD']
    feat_data = pd.DataFrame()
    for i_subj in feat_data_all.index:
        i_subj_data = np.hstack((feat_data_all['w2v2'][i_subj], feat_data_all['hubert'][i_subj]))
        sid, lab = f"{feat_data_all['id'][i_subj]}-{feat_data_all['aug'][i_subj]}", int(feat_data_all['label'][i_subj])
        age, sex = feat_data_all['age'][i_subj], feat_data_all['sex'][i_subj]
        wav = glob.glob(os.path.join(DATA_PATH_PREP,
                                     f"{['HC', 'WD'][lab]}/**/{feat_data_all['id'][i_subj]}_CookieTheft.wav"),
                        recursive=True)[0]
        feat_data = pd.concat([feat_data, pd.DataFrame({'id': [sid], 'age': [age], 'sex': [sex], 
                                                        'label': [lab], 'wav': [wav], 'feat': [i_subj_data]})])
    samp = {'20210616004-1': 1, '20230108044-2': 0}

    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    wav_data_l, cam_att_l, sr = [], [], 16000
    for samp_sid in samp.keys():
        samp_id, samp_aug = int(samp_sid.split("-")[0]), samp_sid.split("-")[-1]
        samp_lab = ['HC', 'WD'][samp[samp_sid]]
        wav_f = feat_data[feat_data['id'] == samp_sid]['wav'].tolist()[0]
        samp_feat = feat_data[feat_data['id'] == samp_sid]['feat'].to_numpy()[0]
        samp_feat = samp_feat.reshape((1, samp_feat.shape[0], samp_feat.shape[-1]))
        pred_lab = ['HC', 'WD'][(model(samp_feat).numpy()[:, 1] > 0.5).astype("int32")[0]]
        print(f"True Label: {samp_lab}; Predicted Label: {pred_lab}")
        if samp_lab != pred_lab:
            continue
        wav_data, sr = librosa.load(wav_f, sr=None, offset=10 * (int(samp_aug.split("_")[0]) - 1), duration=10)
        # wav_data = minmax_scale(wav_data.reshape((1, -1)), (-1, 1), axis=1)[0]
        wav_data_l.append(wav_data)
        score = CategoricalScore([samp[samp_sid]])
        # 获取卷积层的GradCAM：前4个为最后一个Inception模块的1D卷积层，
        # 第5个为InceptionTime模块内部最后一个残差连接中的1D卷积层(若InceptionTime模块个数不是3的整数倍，则没有该项)，
        # 最后一个为InceptionTime模块外部残差连接中的1D卷积层
        conv_l = ['conv1d_22', 'conv1d_23', 'conv1d_24', 'conv1d_25', 'conv1d_26', ]
        cams = []
        for conv in conv_l:
            cams.append(gradcam(score, samp_feat, penultimate_layer=conv))
        # 获取整合跨通道和时间多头注意力机制的网络层激活权重，并求通道上的均值和标准化
        att_wg = get_activations(model, samp_feat, layer_names='concatenate_5')['concatenate_5'][0]
        att_wg_mn = minmax_scale(att_wg.mean(axis=-1).reshape((1, -1)), axis=1)
        cams.append(att_wg_mn)
        cam_att = minmax_scale(np.average(cams, axis=0, weights=[1, 1, 1, 1, 1, 2]), axis=1)[0]
        cam_att_l.append(cam_att)

    idx = 0
    fig, axs = plt.subplots(len(wav_data_l), 1, sharex='col', constrained_layout=True,
                            figsize=(9, 3 * len(wav_data_l)))
    fig.supylabel('Amplitude', fontproperties={'family': font_family, 'size': 18})
    for ts, ca in zip(wav_data_l, cam_att_l):
        interp_function = interp1d(np.arange(ca.shape[-1]), ca, kind='nearest')  # 插值至与音频相同尺寸
        ca = interp_function(np.linspace(0, ca.shape[-1] - 1, ts.shape[-1]))
        t = np.arange(ts.shape[-1])
        points = np.array([t, ts]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(ca.min(), ca.max())
        cmap = plt.get_cmap('Spectral_r')
        lc = LineCollection(segments, cmap=cmap, norm=norm, lw=0.5, ls='solid')
        lc.set_array(ca)
        lc.set_linewidth(0.5)
        col = axs[idx].add_collection(lc)
        axs[idx].set_xlim(t.min(), t.max() + 1)
        axs[idx].set_ylim(ts.min(), ts.max())
        if idx == len(wav_data_l) - 1:
            axs[idx].set_xlabel('Time (s)', fontdict={'family': font_family, 'size': 18})
            cbar = fig.colorbar(col, ax=axs.ravel().tolist(), pad=0.01, shrink=0.95, aspect=25)
            cbar.outline.set_visible(False)
            cbar.ax.set_ylabel('Normalized Weight', fontdict={'family': font_family, 'size': 16})
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.tick_params(length=3)
        axs[idx].set_xticks(np.linspace(0, ts.shape[-1], 6), [str(i) for i in range(0, 11, 2)],
                            fontsize=16, fontproperties=font_family)
        axs[idx].set_yticks(axs[idx].get_yticks(), axs[idx].get_yticklabels(), fontsize=16, fontproperties=font_family)
        axs[idx].yaxis.set_major_locator(plt.MaxNLocator(6))
        axs[idx].yaxis.set_major_formatter(mtick.FuncFormatter(func_format1))
        for sp in axs[idx].spines:
            axs[idx].spines[sp].set_color('k')
            axs[idx].spines[sp].set_linewidth(1)
        axs[idx].tick_params(direction='in', color='k', length=3, width=1)

        t_t = 1600  # 局部图显示权重最大值索引前后各1600个时间点，即1600*2/16000=0.2s
        xlim0, xlim1 = np.argmax(ca) - t_t, np.argmax(ca) + t_t
        ylim0 = np.min(ts[xlim0: xlim1])
        ylim1 = np.max(ts[xlim0: xlim1])
        if np.argmax(ca) < ca.shape[-1] // 2:
            loc = 'lower left'
        else:
            loc = 'lower right'
        axins = inset_axes(axs[idx], width="50%", height="25%", loc=loc, bbox_to_anchor=(0, 0, 1, 1),
                           bbox_transform=axs[idx].transAxes, axes_class=HostAxes)
        p_loc = np.array([np.arange(xlim0, xlim1), ts[xlim0: xlim1]]).T.reshape(-1, 1, 2)
        seg_loc = np.concatenate([p_loc[:-1], p_loc[1:]], axis=1)
        for i, seg in enumerate(seg_loc):
            color = cmap(lc.get_array()[np.argmax(ca) - t_t + i])
            axins.plot(seg[:, 0], seg[:, 1], c=color, lw=0.5)
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        axins.set_xticks([])
        axins.set_yticks([])
        _xlim0, _xlim1 = xlim0 - t_t, xlim1 + t_t
        _ylim0, _ylim1 = ts.min() / 4, ts.max() / 2
        axs[idx].plot([_xlim0, _xlim1, _xlim1, _xlim0, _xlim0], [_ylim0, _ylim0, _ylim1, _ylim1, _ylim0], c='k', lw=1)
        con1 = ConnectionPatch(xyA=(xlim0, ylim1), xyB=(_xlim0, _ylim0), coordsA='data', coordsB='data',
                               axesA=axins, axesB=axs[idx], color='k', lw=1)
        axins.add_artist(con1)
        con2 = ConnectionPatch(xyA=(xlim1, ylim1), xyB=(_xlim1, _ylim0), coordsA='data', coordsB='data',
                               axesA=axins, axesB=axs[idx], color='k', lw=1)
        axins.add_artist(con2)
        for sp in axins.spines:
            axins.spines[sp].set_visible(True)
            axins.spines[sp].set_color('k')
            axins.spines[sp].set_linewidth(1)
        idx += 1
    fig_file = os.path.join(fig_save_dir, f'gradCAM/gradcam.png')
    if not os.path.exists(os.path.dirname(fig_file)):
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02, transparent=True)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    feat_f_wdhc = os.path.join(current_path, r'data/features/features_WDHC.pkl')
    res_path = os.path.join(current_path, r"results")
    pcf_torgo_comp = os.path.join(res_path, r"TORGO_comp.csv")
    pcf_wdhc_comp = os.path.join(res_path, r"WDHC_comp.csv")
    pcf_wdhc_ablation = os.path.join(res_path, r"WDHC_ablation.csv")
    pcf_wdhc_fablation = os.path.join(res_path, r"WDHC_Fablation.csv")
    pcf_wdhc_interaction = os.path.join(res_path, r"WDHC_interaction.csv")
    pcf_wdhc_head = os.path.join(res_path, r"WDHC_head.csv")
    pcf_wdhc_depth = os.path.join(res_path, r"WDHC_depth.csv")

    run_compare_base_flag = False
    if run_compare_base_flag:  # CTCAIT模型与基线比较
        model_d = {"DMFCC-CEEMDAN": BaselineDMFCCModel, "BiLSTM-Attention": BaselineEmbeddingModel,
                   "CTCAIT": CrossTCAInceptionTimeModel, }
        model_compare(feat_f_wdhc, ["PD", "SI"], model_d, pcf_wdhc_comp, mode='w', fit=False, load_data=True)
    run_compare_abla_flag = False
    if run_compare_abla_flag:  # 对CTCAIT模型内部结构的消融
        model_d = {"CTCAIT_dropT": CrossTCAInceptionTimeModel, "CTCAIT_dropC": CrossTCAInceptionTimeModel,
                   "CTCAIT_dropTC": CrossTCAInceptionTimeModel, "CTCAIT_dropR": CrossTCAInceptionTimeModel,
                   "IT": CrossTCAInceptionTimeModel, "CTCA": CrossTCAInceptionTimeModel,
                   "CTCAIT": CrossTCAInceptionTimeModel, }
        model_compare(feat_f_wdhc, ["PD"], model_d, pcf_wdhc_ablation, mode='w', fit=False, load_data=True)
    run_compare_feat_flag = False
    if run_compare_feat_flag:  # 对CTCAIT模型的输入特征消融
        model_d = {"w2v-CTCAIT": CrossTCAInceptionTimeModel, "HuBERT-CTCAIT": CrossTCAInceptionTimeModel,
                   "CTCAIT": CrossTCAInceptionTimeModel, }
        model_compare(feat_f_wdhc, ["PD"], model_d, pcf_wdhc_fablation, mode='w', fit=False, load_data=True)
    run_compare_intera_flag = False
    if run_compare_intera_flag:  # 对CTCAIT模型中跨时间与跨通道交互方式的对比
        model_d = {"CTCAIT_dropTC": LstmInteraInceptionTimeModel, "CTC_LSTM": LstmInteraInceptionTimeModel,
                   "CTCAIT": LstmInteraInceptionTimeModel, }
        model_compare(feat_f_wdhc, ["PD"], model_d, pcf_wdhc_interaction, mode='w', fit=False, load_data=True)
    run_head_flag = False
    if run_head_flag:  # CTCAIT模型头数影响
        mha_head_influ(feat_f_wdhc, ["PD"], pcf_wdhc_head, mode='w', fit=False, load_data=True)
    run_dep_flag = False
    if run_dep_flag:  # CTCAIT模型深度影响
        depth_influ(feat_f_wdhc, ["PD"], pcf_wdhc_depth, mode='w', fit=False, load_data=True)
    run_cd_flag = True
    if run_cd_flag:  # 多模型多数据集统计比较CD图：Critical Difference Diagrams
        pd_md_com = pd.concat([pd.read_csv(pcf_wdhc_comp), pd.read_csv(pcf_torgo_comp)], ignore_index=True)
        cd_diagram(md_com_data=pd_md_com, fig_save_dir=res_path, datasets_col='Task',
                   models_col='Feature-Model Config', metrics_col='Acc/%')
    run_exp_flag = False
    if run_exp_flag:  # 模型预测解释
        gradcam_vis(os.path.join(current_path, r'results'))

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(os.path.join(current_path, r"results/finished.txt"), "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
