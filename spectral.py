import torch
import scipy.sparse as scisparse
from scipy.sparse import linalg as sla

def calc_tri_areas(vertices, faces):
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]
    
    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5


def calc_LB_FEM(vertices, faces, device='cuda'):
    n = vertices.shape[0]
    m = faces.shape[0]
    
    angles = {}
    for i in (1.0, 2.0, 3.0):
        a = torch.fmod(torch.as_tensor(i - 1), torch.as_tensor(3.)).long()
        b = torch.fmod(torch.as_tensor(i), torch.as_tensor(3.)).long()
        c = torch.fmod(torch.as_tensor(i + 1), torch.as_tensor(3.)).long()

        ab = vertices[faces[:,b],:] - vertices[faces[:,a],:];
        ac = vertices[faces[:,c],:] - vertices[faces[:,a],:];

        ab = torch.nn.functional.normalize(ab, p=2, dim=1)
        ac = torch.nn.functional.normalize(ac, p=2, dim=1)
        
        o = torch.mul(ab, ac)
        o = torch.sum(o, dim=1)
        o = torch.acos(o)
        o = torch.div(torch.cos(o), torch.sin(o))
        
        angles[i] = o
    
    indicesI = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 1], faces[:, 0]))
    indicesJ = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2]))
    indices = torch.stack((indicesI, indicesJ))
    
    one_to_n = torch.arange(0, n, dtype=torch.long, device=device)
    eye_indices = torch.stack((one_to_n, one_to_n))

    values = torch.cat((angles[3], angles[1], angles[2], angles[1], angles[3], angles[2])) * 0.5

    
    stiff = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                                 values=-values,
                                 device=device,
                                 size=(n, n)).coalesce()
    stiff = stiff + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                 values=-torch.sparse.sum(stiff, dim=0).to_dense(),
                                 device=device,
                                 size=(n, n)).coalesce()
    
    areas = calc_tri_areas(vertices, faces)
    areas = areas.repeat(6) / 12.
    
    mass = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                             values=areas,
                             device=device,
                             size=(n, n)).coalesce()
    mass = mass + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                 values=torch.sparse.sum(mass, dim=0).to_dense(),
                                 device=device,
                                 size=(n, n)).coalesce()
    
    lumped_mass = torch.sparse.sum(mass, dim=1).to_dense()

    return stiff, mass, lumped_mass

def sparse_dense_mul(s, d):
    # implements point-wise product sparse * dense
    s = s.coalesce()
    i = s.indices()
    v = s.values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size()).coalesce()

def decomposition_torch(stiff, lumped_mass):
    # Cholesky decomposition for diagonal matrices
    lower = torch.sqrt(lumped_mass)

    # Compute inverse
    lower_inv = 1 / lower

    # todo1: when pytorch will support broadcastin on sparse tensor it will be enough:
    # C = lower_inv[None, :] * stiff * lower_inv[:, None]
    #
    # todo2: in alternative, use sparse @ stiff @ sparse when supported
    C = sparse_dense_mul(stiff, lower_inv[None, :] * lower_inv[:, None])  # <- INEFFICIENCY
    return C

def eigsh(A, values, indices, k, sigma=-1e-5):
    device = A.device
    precision = A.dtype

    values = values.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()
                
    Ascipy = scisparse.coo_matrix((values, indices)).tocsc()

    e, phi = sla.eigsh(Ascipy, k, sigma=sigma)

    return e, phi
