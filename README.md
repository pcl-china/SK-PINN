# SK-PINN
SK-PINN: Smoothing kernel gradients to accelerate physics-informed neural networks training

A smoothing kernel-based physics-informed neural network (SK-PINN) which is combined with the smoothed particle hydrodynamics (SPH) for the discrete differential equation is proposed in this paper. SK-PINN provides an accelerated training method for physics-informed deep learning which can break the computational bottlenecks associated with automatic differentiation (AD) in conventional PINNs. It creases a robust framework capable of solving problems in complex solution domains and computational mechanics. When the number of collocation points gradually increases, the training speed of SK-PINN significantly surpasses that of conventional PINNs. Additionally, we compare the convergence properties of SK-PINN with the conventional PINN by using the neural tangent kernel (NTK) theory. Finally, the effectiveness of our proposed SK-PINN is demonstrated through several examples involving regular and complex domains, as well as both forward and inverse problems in fluid dynamics and solid mechanics.

## The manuscript is being revised and will go to the submission stage.
## If the manuscript is accepted the code will be made public.
