This repository is associated with the bachelor's thesis:

# DirectTransEst: Efficient Transferability Estimation for Domain Adaption in Medical Image Segmentation
Author: Johanna Schlimme \
Supervisors: Prof. Dr. Christoph Lippert (Chair of Digital Health and Machine Learning), Alexander Rakowski (Research Assistant) \
Hasso Plattner Institute at University of Potsdam, August 2023

## Thesis Abstract
Transfer learning offers a promising approach for deep learning in medical image segmentation due to the limited availability of labeled data. However, assessing whether a pre-trained model will effectively adapt to a new target dataset remains challenging. Current methodologies to estimate this transferability are intricate, reliant on source data, or insufficient for the unique attributes of medical images. This study investigates: "Can the transferability of pre-trained medical models for domain adaption be estimated based on the performance of the respective pre-trained model before fine-tuning?" We introduced DirectTransEst, a technique that estimates transferability by evaluating the source model's performance directly on the target dataset before fine-tuning. We trained models on four datasets using the residual U-Net architecture, emphasizing three brain segmentation tasks. After initial training, we assessed each model's transferability and fine-tuned them. Our findings showed that the performance scores obtained before fine-tuning strongly correlated with those achieved after fine-tuning. Based on this high correlation, DirectTransEst surpassed the performance of two established Transferability Estimation (TE) methods we compared it against. Due to its straightforward nature and lack of dependency on source data, DirectTransEst presents itself as an efficient TE technique for domain adaptation in medical image segmentation. Future research should explore its applicability across varied datasets, tasks, and architectures.

## Code References
CC-FV: \
Yang, Y., Wei, M., He, J., Yang, J., Ye, J., & Gu, Y. (2023). Pick the best pre-trained model: Towards transferability estimation for medical image segmentation. arXiv preprint arXiv:2307.11958. \
TransRate: \
Huang, L., Huang, J., Rong, Y., Yang, Q. &amp; Wei, Y.. (2022). Frustratingly Easy Transferability Estimation. <i>Proceedings of the 39th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 162:9201-9225 Available from https://proceedings.mlr.press/v162/huang22d.html.

