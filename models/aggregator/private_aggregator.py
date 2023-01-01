from .aggregator import AggregatorFactory, Aggregator
from utils import gradient_voting_rdp, gradient_voting_rdp_multiproj, ComputeDPPrincipalProjection
from sklearn.random_projection import GaussianRandomProjection
import torch
from typing import List
import numpy as np

@AggregatorFactory.register('private_aggregator')
class PrivateAggregator(Aggregator):
    def __init__(self, **kwargs) -> None:
        self.step_size = 1e-4
        
        self.random_proj = True
        self.proj_mat = 1

        self.pca = False
        self.pca_sigma = 1.0
        self.pca_dim = 10
        
        self.sigma = 2000
        self.sigma_thresh = 4500
        self.thresh = 0.5

        self.data_dim = [1, 28, 28]
        self.dp_delta = 1e-5

        # if kwargs['orders'] is not None:
        #     self.orders = np.asarray(kwargs['orders'])
        # else:
            
        self.orders = np.hstack([1.1, np.arange(2, 200)])

        self.rdp_counter = np.zeros(self.orders.shape)
        
        if self.pca:
            data = kwargs['data_X'].reshape([kwargs['data_X'].shape[0], -1])
            self.pca_components, rdp_budget = ComputeDPPrincipalProjection(
                data,
                self.pca_dim,
                self.orders,
                self.pca_sigma,
            )
            self.rdp_counter += rdp_budget

    

    def _aggregate_results(self, output_list, epoch):
        if self.pca:
            res, rdp_budget = gradient_voting_rdp(
                output_list,
                self.step_size,
                self.sigma,
                self.sigma_thresh,
                self.orders,
                pca_mat=self.pca_components,
                thresh=self.thresh
            )
        elif self.random_proj:

            proj_dim = self.pca_dim
            if epoch is not None:
                proj_dim = min(epoch + 1, self.pca_dim)
            else:
                proj_dim = self.pca_dim
            n_data = len(output_list)
            orig_dim = output_list[0].shape[0]

            if self.proj_mat > 1:
                proj_dim_ = proj_dim // self.proj_mat
                n_data_ = n_data // self.proj_mat
                orig_dim_ = orig_dim // self.proj_mat
                print("n_data:", n_data)
                print("orig_dim:", orig_dim)
                transformers = [GaussianRandomProjection(n_components=proj_dim_) for _ in range(self.proj_mat)]
                for transformer in transformers:
                    transformer.fit(np.zeros([n_data_, orig_dim_]))
                    # print(transformer.components_.shape)
                proj_matrices = [np.transpose(transformer.components_) for transformer in transformers]
                res, rdp_budget = gradient_voting_rdp_multiproj(
                    output_list,
                    self.step_size,
                    self.sigma,
                    self.sigma_thresh,
                    self.orders,
                    pca_mats=proj_matrices,
                    thresh=self.thresh
                )
            else:
                transformer = GaussianRandomProjection(n_components=proj_dim)
                transformer.fit(np.zeros([n_data, orig_dim]))  # only the shape of output_list[0] is used
                proj_matrix = np.transpose(transformer.components_)

            # proj_matrix = np.random.normal(loc=np.zeros([orig_dim, proj_dim]), scale=1/float(proj_dim), size=[orig_dim, proj_dim])
                res, rdp_budget = gradient_voting_rdp(
                    output_list,
                    self.step_size,
                    self.sigma,
                    self.sigma_thresh,
                    self.orders,
                    pca_mat=proj_matrix,
                    thresh=self.thresh)
        else:
            res, rdp_budget = gradient_voting_rdp(
                    output_list,
                    self.step_size,
                    self.sigma,
                    self.sigma_thresh,
                    self.orders,
                    thresh=self.thresh
                )
        return res, rdp_budget

    def aggregate(self, grads_list: List[torch.Tensor], epoch) -> torch.Tensor:
        aggregated_grads = []
        batch_size = grads_list[0].shape[0]
        device = grads_list[0].device
        
        for j in range(batch_size):
            batch_grads = [grads[j] for grads in grads_list]
            aggregated_grad, rdp_budget = self._aggregate_results(batch_grads, epoch)
            aggregated_grads.append(aggregated_grad)
            self.rdp_counter += rdp_budget

        perturbation = torch.from_numpy(np.vstack(aggregated_grads)).float().to(device)
        return perturbation
