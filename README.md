# On the Importance of Image Encoding in Automated Chest X-Ray Report Generation
### BMVC, 2022

The reference code of [On the Importance of Image Encoding in Automated Chest X-Ray Report Generation](https://arxiv.org/abs/2211.13465).

The original code taken from [Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation](https://github.com/ysmiura/ifcc).

Changes to original code:
- [custom_models.py](custom_models.py) - contains various image encoders
- [image.py](clinicgen/models/image.py) - modified code of original file to accomodate various encoders

### Running the code
The original [repository](https://github.com/ysmiura/ifcc) has very good instructions for running the code. We recommend following those.
To choose the encoders please select corresponding models within the [custom_models.py](custom_models.py) file.
New encoders can also be easily added.

## Licence
See LICENSE and clinicgen/external/LICENSE_bleu-cider-rouge-spice for details.
