# baby diffusion model

Yet another implementation of the DDPM paper: https://arxiv.org/abs/2006.11239

#### A DDPM is pretty much this

$$
\begin{array}{|c|c|}
\hline \text { Training } & \text { Sampling } \\
\hline \begin{array}{l}
\textbf{Input}: \text {Training data } \mathbf{x} \\
\textbf{Output}: \text { Model parameters } \boldsymbol{\phi}_t \\
\textbf{repeat} \\
\quad  \textbf { for } i \in \mathcal{B} \textbf { do} \\
\quad \quad t \sim \text {Uniform} [1, \ldots T] \\
\quad \quad \boldsymbol{\epsilon} \sim \operatorname{Norm}[\mathbf{0}, \mathbf{I}] \\
\quad \quad \ell_i=\left\|\mathbf{g}_t\left[\sqrt{\alpha_t} \mathbf{x}_i+\sqrt{1-\alpha_t} \boldsymbol{\epsilon}, \phi_t\right]-\boldsymbol{\epsilon}\right\|^2 \\
\quad \text {Take gradient descent step} \\
\textbf{until} \text { converged} \\
\end{array} & \begin{array}{l}
\textbf{Input}: \text {Model}, \mathbf{g}_t\left[\bullet, \boldsymbol{\phi}_t\right] \\
\textbf{Output}: \text {Sample}, \mathbf{x} \\
\mathbf{z}_T \sim \operatorname{Norm}_{\mathbf{z}}[\mathbf{0}, \mathbf{I}] \\
\textbf{for } t=T... 2 \textbf{ do} \\
\quad \hat{\mathbf{z}}_{t-1}=\frac{1}{\sqrt{1-\beta_t}} \mathbf{z}_t-\frac{\beta_t}{\sqrt{1-\alpha_t} \sqrt{1-\beta_t}} \mathbf{g}_t\left[\mathbf{z}_t, \boldsymbol{\phi}_t\right] \\
\quad \boldsymbol{\epsilon} \sim \operatorname{Norm}_{\boldsymbol{\epsilon}}[\mathbf{0}, \mathbf{I}] \\
\quad \mathbf{z}_{t-1}=\hat{\mathbf{z}}_{t-1}+\sigma_t \boldsymbol{\epsilon} \\
\mathbf{x}=\frac{1}{\sqrt{1-\beta_1}} \mathbf{z}_1-\frac{\beta_1}{\sqrt{1-\alpha_1} \sqrt{1-\beta_1}} \mathbf{g}_1\left[\mathbf{z}_1, \boldsymbol{\phi}_1\right] \\
\end{array} \\
\hline
\end{array}
$$