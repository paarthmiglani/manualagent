import torch
import numpy as np

class ActiveLearner:
    def __init__(self, model, unlabeled_data_loader):
        self.model = model
        self.unlabeled_data_loader = unlabeled_data_loader

    def query(self, n_instances):
        """
        Query the unlabeled data for the most informative instances to label.
        """
        self.model.eval()
        uncertainties = []
        with torch.no_grad():
            for images, _, _, _ in self.unlabeled_data_loader:
                log_probs = self.model(images)
                # Use entropy as a measure of uncertainty
                entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=2)
                uncertainties.extend(entropy.mean(dim=0).tolist())

        # Select the most uncertain instances
        uncertain_indices = np.argsort(uncertainties)[-n_instances:]
        return uncertain_indices
