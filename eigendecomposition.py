import scipy.sparse as scisparse
from scipy.sparse import linalg as sla
import torch

class Eigsh_torch(torch.autograd.Function):
    # Reference issue
    # https://github.com/pytorch/pytorch/issues/24185

    @staticmethod
    def forward(ctx, A, values=None, indices=None, k=None, sigma=-1e-5, maxiter=None, tol=0):
        # Currently using scipy, on cpu, as backend
        # (torch.symeig computes all eigenvalues only on dense tensors)
        #
        # todo: check if CuPy will support eigsh:
        # https://docs-cupy.chainer.org/en/stable/reference/comparison.html?highlight=eigsh
        assert values is not None
        assert indices is not None
        assert k is not None

        device = A.device
        precision = A.dtype

        values = values.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
                
        Ascipy = scisparse.coo_matrix((values, indices)).tocsc()

        e, phi = sla.eigsh(
            Ascipy,
            k,
            sigma=sigma)

        e, phi = (torch.tensor(e, dtype=precision, device=device, requires_grad=False),
                  torch.tensor(phi, dtype=precision, device=device, requires_grad=False))

        ctx.save_for_backward(phi)
        return e

    @staticmethod
    def backward(ctx, glambda):
        # Implementation reference:
        # https://github.com/pytorch/pytorch/blob/9228dd766af09017364bfaa7f88feacb7e89a154/tools/autograd/templates/Functions.cpp#L1701
        # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
        v, = ctx.saved_tensors
        device = glambda.device
        # v:                   [n, k]
        # glambda:             [k]
        # result:              [n, n]
        vt = v.transpose(-2, -1)
        
        # v @ diag(glambda) @ vt
        result = (v * glambda[None, :]) @ vt

        return result, None, None, None, None, None