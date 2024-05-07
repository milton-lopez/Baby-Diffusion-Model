# baby diffusion model

Yet another implementation of the DDPM paper: https://arxiv.org/abs/2006.11239

## A DDPM is pretty much this:

**Training:**
- **Input**: Training data $\mathbf{x}$
- **Output**: Model parameters $\phi_t$
- **Repeat**:
    - **For** $i \in \mathcal{B}$ **do**:
        - $t \sim \text{Uniform}[1, \ldots, T]$
        - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        - $\ell_i = \left\|\mathbf{g}_t\left(\sqrt{\alpha_t} \mathbf{x}_i + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}, \phi_t\right) - \boldsymbol{\epsilon}\right\|^2$
    - Accumulate losses for batch and take gradient step
- **Until** converged

**Sampling:**
- **Input**: Model, $\mathbf{g}_t(\cdot, \phi_t)$
- **Output**: Sample, $\mathbf{x}$
- $\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- **For** $t = T \ldots 2$ **do**:
    - $\hat{\mathbf{z}}_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \alpha_t} \sqrt{1 - \beta_t}} \mathbf{g}_t(\mathbf{z}_t, \boldsymbol{\phi}_t)$
    - $\boldsymbol{\epsilon} \sim \mathcal{N}_\epsilon(\mathbf{0}, \mathbf{I})$
    - $z_{t-1} = \hat{z}_{t-1} + \sigma_t \epsilon$
- $\mathbf{x} = \frac{1}{\sqrt{1 - \beta_1}} \mathbf{z}_1 - \frac{\beta_1}{\sqrt{1 - \alpha_1} \sqrt{1 - \beta_1}} \mathbf{g}_1(\mathbf{z}_1, \phi_1)$

**TODO**:
- [ ] More toy synth datasets
