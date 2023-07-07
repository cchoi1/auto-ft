from torch.utils.data import Dataset
class TaskDataset(Dataset):
    def __init__(self, train_dataset, test_dataset, meta_batch_size, batch_size, device=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.dataset = dataset
        self.device = device
        self.meta_batch_size = meta_batch_size
        self.batch_size = batch_size

        # NOTE: do not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 20

        # save pointer of current read batch in total cache
        self.indexes = {'train': 0, 'test': 0}
        self.datasets = {
            'train': self.x_train,
            'test': self.x_test,
        }  # original data cached
        print('DB: train', self.x_train.shape, 'test', self.x_test.shape)

        self.datasets_cache = {
            'train': self.load_data_cache(self.datasets['train']),  # current epoch data cached
            'test': self.load_data_cache(self.datasets['test']),
        }

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """

        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        for _sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for _ in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = self.rng.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):
                    selected_img = self.rng.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[: self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot :]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = self.rng.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(
                    self.n_way * self.k_shot,
                    1,
                    self.resize,
                    self.resize,
                )[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = self.rng.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(
                    self.n_way * self.k_query,
                    1,
                    self.resize,
                    self.resize,
                )[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            x_spts = np.array(x_spts, dtype=np.float32).reshape(
                self.batchsz,
                setsz,
                1,
                self.resize,
                self.resize,
            )  # [b, setsz, 1, 84, 84]
            y_spts = np.array(y_spts, dtype=np.int).reshape(
                self.batchsz,
                setsz,
            )  # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys, dtype=np.float32).reshape(
                self.batchsz,
                querysz,
                1,
                self.resize,
                self.resize,
            )
            y_qrys = np.array(y_qrys, dtype=np.int).reshape(self.batchsz, querysz)

            x_spts, y_spts, x_qrys, y_qrys = (
                torch.from_numpy(z).to(self.device) for z in [x_spts, y_spts, x_qrys, y_qrys]
            )

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """

        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch