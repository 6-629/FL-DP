import models, torch, copy
from tqdm import tqdm
import numpy as np


class Client(object):

    def __init__(self, conf, model, train_dataset, id=-1):
        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.client_id = id
        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(tqdm(self.train_loader)):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            print("Epoch %d done." % e)

        # Compute model update with added noise for differential privacy
        diff = dict()
        noise_type = self.conf.get('dp_noise_type', 'gaussian')  # Select noise type
        noise_scale = self.conf.get('dp_noise_scale', 0.1)  # Set noise scale for DP

        for name, data in self.local_model.state_dict().items():
            if noise_type == 'gaussian':
                # Add Gaussian noise
                noise = torch.normal(0, noise_scale, size=data.size(), device=data.device)
            elif noise_type == 'laplace':
                # Add Laplace noise
                noise = torch.from_numpy(np.random.laplace(0, noise_scale, data.size())).float().to(data.device)
            elif noise_type == 'gradient_clipping':
                # Apply gradient clipping and then add Gaussian noise
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                noise = torch.normal(0, noise_scale, size=data.size(), device=data.device)
            else:
                raise ValueError("Unknown noise type")

            diff[name] = (data - model.state_dict()[name]) + noise  # Add noise after computing the update

        return diff
