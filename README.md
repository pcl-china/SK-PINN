# SK-PINN
SK-PINN: Accelerated physics-informed deep learning by smoothing kernel gradients

The automatic differentiation (AD) in the vanilla physics-informed neural networks (PINNs) is the computational bottleneck for the high-efficiency analysis. The concept of derivative discretization in smoothed particle hydrodynamics (SPH) can provide an accelerated training method for PINNs. In this paper, smoothing kernel physics-informed neural networks (SK-PINNs) are established, which solve differential equations using smoothing kernel discretization. It is a robust framework capable of solving problems in the computational mechanics of complex domains. When the number of collocation points gradually increases, the training speed of SK-PINNs significantly surpasses that of vanilla PINNs. In cases involving large collocation point sets or higher-order problems, SK-PINN training can be up to tens of times faster than vanilla PINN. Additionally, analysis using neural tangent kernel (NTK) theory shows that the convergence rates of SK-PINNs are consistent with those of vanilla PINNs. The superior performance of SK-PINNs is demonstrated through various examples, including regular and complex domains, as well as forward and inverse problems in fluid dynamics and solid mechanics.

## The manuscript has been submitted and is now under review.
## If the manuscript is accepted the code will be made public.
# arXiv link：
https://arxiv.org/abs/2411.02411
