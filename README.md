# From-Image-to-Imuge-Immunized-Image-Generation

Official code of the ACMMM 2021 accepted paper.

Digital images are vulnerable to nefarious tampering attacks such as content addition or removal that severely alters the original meaning. It is somehow like a person without protection that is open to various kinds of viruses. Image immunization is a technology of protecting the images by introducing trivial perturbation, so that the protected images are immune to the viruses in that the tampered contents can be auto-recovered.This paper presents an enhanced network for image immunization. By observing the invertible relationship between image immunization and the corresponding self-recovery, we employ an invertible neural network to jointly learn image immunization and recovery respectively in the forward and backward pass. We also introduce an efficient attack layer that involves both malicious tamper and benign image post-processing, where a novel distillation-based JPEG simulator is proposed for improved JPEG robustness.  Our method achieves promising results in real-world tests where experiments show accurate tamper localization as well as high-fidelity content recovery. Additionally, we show superior performance on tamper localization compared to state-of-the-art schemes based on passive forensics.

We have just released the source code of the submitted supplementary material to ACMMM 2021. Feel free to contact me via shinydotcom@163.com if you encounter any problem during using this project.