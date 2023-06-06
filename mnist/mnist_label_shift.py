"""Adapted from https://github.com/Angie-Liu/labelshift."""

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from torchvision import datasets


class MNISTLabelShift(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, root, training_size, testing_size, shift_type, parameter, parameter_aux=None, target_label=None,
                 train=True, transform=None, target_transform=None, download=False):

        """
        Converted function from original torch mnist class :
        new parameters:
        training_size: int, sample size of training
        testing_size: int, sample size of both training and testing set
        shift_type: int, 1 for knock one shift on source, target is uniform
                         2 for tweak one shift on source, target is uniform
                         3 for dirichlet shift on source, target is uniform
                         4 for dirichlet shift on the target set, source is uniform
                         5 for tweak one shift on the target set, source is uniform
                         6 for Minority class shift on source, target is uniform
                         7 for tweak on shift on both source and target
        parameter: float in [0, 1], delta for knock one shift, delete target_label by delta
                                    or, rho for tweak one shift, set target_label probability as rho, others even
                                    or, alpha for dirichlet shift
                                    or, # of minority class for Minority class shift
                                    or, rho for tweak one shift on source for 7
        parameter_aux: float in [0, 1], required for 6 and 7
                                    6, the proportion of data for minority class
                                    7, rho for tweak one shift on target for 7

        target_label: int, target label for knock one and tweak one shift
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        features, labels = torch.load(os.path.join(self.root, self.processed_folder, data_file))

        features = features.numpy()
        labels = labels.numpy()
        indices = np.random.permutation(60000)
        if shift_type == 1 or shift_type == 2 or shift_type == 3 or shift_type == 6:

            m_test = testing_size

            test_indices = indices[0: m_test]
            train_indices = indices[m_test:]
        elif shift_type != 7:
            m_train = training_size

            train_indices = indices[0: m_train]
            test_indices = indices[m_train:]
        else:
            m_train = 30000
            m_test = 30000
            train_indices = indices[0: m_train]
            test_indices = indices[m_train:]

        test_data = features[(test_indices,)]
        test_labels = labels[(test_indices,)]

        train_data = features[(train_indices,)]
        train_labels = labels[(train_indices,)]

        if shift_type == 1:
            if target_label == None:
                raise RuntimeError("There should be a target label for the knock one shift.")
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            num_knock = int(num_target * parameter)
            train_data = np.delete(train_data, indices_target[0:num_knock], 0)
            train_labels = np.delete(train_labels, indices_target[0:num_knock])
            m_train = len(train_labels)

        elif shift_type == 2:
            if target_label == None:
                raise RuntimeError("There should be a target label for the tweak one shift.")
            # use the number of target label to decide the total number of the training samples

            if parameter < (1.0 - parameter) / 9:
                target_label = (target_label + 1) % 10
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            num_train = int(num_target / parameter)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)

            num_remain = num_train - num_target
            # even on other labels
            num_i = int(num_remain / 9)
            indices_train = np.empty((0, 1), dtype=int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i]
                else:
                    indices_i = indices_i[0:num_target]
                indices_train = np.append(indices_train, indices_i)

            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)

        elif shift_type == 3:
            # use the maximum prob to decide the total number of training samples
            target_label = np.argmax(parameter)
            print(parameter)

            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            prob_max = np.amax(parameter)
            num_train = int(num_target / prob_max)
            indices_train = np.empty((0, 1), dtype=int)
            for i in range(10):
                num_i = int(num_train * parameter[i])
                indices_i = np.where(train_labels == i)[0]
                indices_i = indices_i[0:num_i]
                indices_train = np.append(indices_train, indices_i)

            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)

        elif shift_type == 4:

            # use the maximum prob to decide the total number of training samples
            target_label = np.argmax(parameter)
            print('Dirichlet shift with prob,', parameter)

            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)
            prob_max = np.amax(parameter)
            m_test = int(num_target / prob_max)
            indices_test = np.empty((0, 1), dtype=int)

            for i in range(10):
                num_i = int(m_test * parameter[i])
                indices_i = np.where(test_labels == i)[0]
                indices_i = indices_i[0:num_i]
                indices_test = np.append(indices_test, indices_i)
            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)

        elif shift_type == 5:
            if target_label == None:
                raise RuntimeError("There should be a target label for the tweak one shift.")
            # use the number of target label to decide the total number of the training samples

            if parameter < (1.0 - parameter) / 9:
                target_label = (target_label + 1) % 10
            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)
            num_test = int(num_target / parameter)

            num_remain = num_test - num_target
            # even on other labels
            num_i = int(num_remain / 9)
            indices_test = np.empty((0, 1), dtype=int)

            for i in range(10):
                indices_i = np.where(test_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i]
                else:
                    indices_i = indices_i[0:num_target]
                indices_test = np.append(indices_test, indices_i)

            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)

        elif shift_type == 6:
            # randomly choose 2 target labels
            target_label = np.linspace(1, parameter, parameter)
            para = parameter_aux
            prob = (1 - len(target_label) * para) / (10 - len(target_label))
            indices_target = np.where(train_labels == target_label[0])[0]
            num_target = len(indices_target)
            num_train = int(num_target / prob)
            num_target = int(num_train * prob)

            # even on other labels
            num_i = int(num_train * para)
            indices_train = np.empty((0, 1), dtype=int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i in target_label:
                    indices_i = indices_i[0:num_i]
                else:
                    indices_i = indices_i[0:num_target]
                indices_train = np.append(indices_train, indices_i)

            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)
        elif shift_type == 7:

            if parameter < (1.0 - parameter) / 9:
                target_label = (target_label + 1) % 10
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            num_train = int(num_target / parameter)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)

            num_remain = num_train - num_target
            # even on other labels
            num_i = int(num_remain / 9)
            indices_train = np.empty((0, 1), dtype=int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i]
                else:
                    indices_i = indices_i[0:num_target]
                indices_train = np.append(indices_train, indices_i)

            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)
            print(m_train)
            # testint portion

            if parameter_aux < (1.0 - parameter_aux) / 9:
                target_label = (target_label + 1) % 10
            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)
            num_test = int(num_target / parameter_aux)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)

            num_remain = num_test - num_target
            # even on other labels
            num_i = int(num_remain / 9)
            indices_test = np.empty((0, 1), dtype=int)
            print(num_i)
            for i in range(10):
                indices_i = np.where(test_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i]
                else:
                    indices_i = indices_i[0:num_target]
                indices_test = np.append(indices_test, indices_i)

            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)


        else:
            raise RuntimeError("Invalid shift type.")

        if m_test < testing_size:
            testing_size = m_test
        if m_train < training_size:
            training_size = m_train

        train_data = train_data[range(training_size)]
        train_labels = train_labels[range(training_size)]

        test_data = test_data[range(testing_size)]
        test_labels = test_labels[range(testing_size)]

        self.data = torch.from_numpy(np.concatenate((test_data, train_data)))
        self.labels = torch.from_numpy(np.concatenate((test_labels, train_labels)))
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.m_test = testing_size
        self.m_train = training_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_train_label(self):
        return self.train_labels

    def get_test_label(self):
        return self.test_labels

    def get_testsize(self):
        return self.m_test

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            datasets.utils.download_url(url, root=os.path.join(self.root, self.raw_folder),
                                        filename=filename, md5=None)
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)