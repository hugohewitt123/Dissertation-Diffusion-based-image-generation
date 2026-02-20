# Official Implementation of DOODL (End-to-End Diffusion Latent Optimization Improves Classifier Guidance)

[Arxiv](https://arxiv.org/abs/2303.13703)



# What is DOODL?
From the original authors:

DOODL (Direct Optimization of Diffusion Latents) is a variant of classifier guidance that directly optimizes diffusion latents `x_T` instead of using model-based gradients to guide denoising. This is done be leveraging the [EDICT](https://arxiv.org/abs/2211.12446) algorithm and [MemCNN](https://github.com/silvandeleemput/memcnn) library to construct a diffusion process that can be backpropagated through with constant memory cost w.r.t the number of diffusion steps without significant runtime increase. The control of this optimization allows a variety of guidance modes to be incorporated. See the paper: [paper](https://arxiv.org/abs/2303.13703) for more information.


# BDIA_DOODL

This is an implementation of DOODL that doesn't used EDCIT algorithm, instead this system used the BDIA method

# Setup

## HF Auth token

Paste a copy of a suitable [HF Auth Token](https://huggingface.co/docs/hub/security-tokens) into [hf_auth](hf_auth) with no new line (to be read by the following code in `edict_functions.py`)
```
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()
    
```

Example file at `./hf_auth`
```
abc123abc123
```

# Code structure


"BDIA_DOODL" contatains the BDIA implementation

"doodl" contains the original code for reference and comparing results

"helper_functions" contains the functionality of DDIM and functions to evaluate alpha values

The jupyter file "BDIADOODLexamples" contains the functionality to run the code, this is the file to run for BDIA_DOODL results.
It may be nescessary to run the code in Google Colab because of the GPU requirements of the system.

