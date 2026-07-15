from .pinn import PINN, ParametricPINN, predict
from .losses import (
    pde_loss, pde_loss_multi_re,
    data_loss, data_loss_full,
    total_loss, total_loss_cavity, total_loss_cavity_multi_re,
    bc_loss_inflow, bc_loss_outlet, bc_loss_walls, bc_loss_cylinder,
    bc_loss_cavity, bc_loss_cavity_walls, bc_loss_cavity_lid,
    _on_device,
)
