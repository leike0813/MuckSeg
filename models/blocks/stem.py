import torch
import torch.nn as nn
from .auxiliaries import LayerNorm


class StemBlock(nn.Module):
    """Stem block for muck segmentation task.

       All right preserved by Joshua Reed.

    Args:
        in_chans (int): Number of input channels.
        dim (int): Number of output channels. This will greatly affect the model performance and computational efficiency
        routes (Sequence of str): Routes to be used. choose from AVAILABLE_ROUTES
        mlp_ratio (float): Expansion ratio of hidden feature dimension in MLP layer. Default: 4.0
    """
    AVAILABLE_ROUTES = {
        'RAW': lambda x: nn.Identity(),
        'MAXPOOL': lambda x: nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        'AVGPOOL': lambda x: nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        '3CONV': lambda x: nn.Conv2d(x, x, kernel_size=3, padding=1, groups=x, bias=False),
        '5CONV': lambda x: nn.Conv2d(x, x, kernel_size=5, padding=2, groups=x, bias=False),
        '7CONV': lambda x: nn.Conv2d(x, x, kernel_size=7, padding=3, groups=x, bias=False),
        '9CONV': lambda x: nn.Conv2d(x, x, kernel_size=9, padding=4, groups=x, bias=False),
        'D-3CONV': lambda x: nn.Sequential(*[nn.Conv2d(x, x, kernel_size=3, padding=1, groups=x, bias=False) for i in range(2)]),
        'D-5CONV': lambda x: nn.Sequential(*[nn.Conv2d(x, x, kernel_size=5, padding=2, groups=x, bias=False) for i in range(2)]),
        'T-3CONV': lambda x: nn.Sequential(*[nn.Conv2d(x, x, kernel_size=3, padding=1, groups=x, bias=False) for i in range(3)]),
        'A-3CONV': lambda x: nn.Conv2d(x, x, kernel_size=3, padding=2, dilation=2, groups=x, bias=False),
        'A-5CONV': lambda x: nn.Conv2d(x, x, kernel_size=5, padding=4, dilation=2, groups=x, bias=False),
    }
    DEFAULT_ROUTES = {
        'RAW': 1,
        'MAXPOOL': 2,
        'AVGPOOL': 1,
    }

    def __init__(self, in_chans, dim, routes, mlp_ratio=4.):
        super(StemBlock, self).__init__()
        self.dim = dim
        self.pwconv = nn.Conv2d(in_chans, dim, kernel_size=1, padding=0, bias=False)

        for route in routes:
            assert route.upper() in StemBlock.AVAILABLE_ROUTES.keys(),\
                'Invalid route name: {}'.format(route)
        routes = set([route.upper() for route in routes])
        self.route_names = []
        self.route_dims = []
        for default_route in StemBlock.DEFAULT_ROUTES.keys(): # add 'RAW', 'MAXPOOL', 'AVGPOOL'
            if default_route in routes:
                routes.discard(default_route)
            self.route_names.append(default_route)
            self.route_dims.append(StemBlock.DEFAULT_ROUTES[default_route])
        # add extra routes
        extra_routes = []
        for available_route in StemBlock.AVAILABLE_ROUTES.keys():
            if available_route in routes:
                extra_routes.append(available_route)
        n_extra_routes = len(extra_routes)
        extra_route_dims = [(dim - 4) // n_extra_routes for i in range(n_extra_routes)]
        extra_route_dims[-1] += (dim - 4) % n_extra_routes

        self.route_names.extend(extra_routes)
        self.route_dims.extend(extra_route_dims)
        self.n_routes = len(self.route_names)

        self.routes = nn.ModuleList()
        for i, route in enumerate(self.route_names):
            self.routes.append(StemBlock.AVAILABLE_ROUTES[route](self.route_dims[i]))

        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.linear1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.linear2 = nn.Linear(int(mlp_ratio * dim), dim)

    def forward(self, x):
        x = self.pwconv(x)
        x = torch.split(x, self.route_dims, dim=1)
        x = torch.cat([route(x[i]) for i, route in enumerate(self.routes)], dim=1)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = x.permute(0, 3, 1, 2)

        return x

    def init_maxpool_route_weight(self):
        _max_weight = torch.max(torch.abs(self.pwconv.weight)).detach()
        self.pwconv.weight.data[0, 0, 0, 0] = _max_weight # RAW
        self.pwconv.weight.data[1, 0, 0, 0] = _max_weight  # MAXPOOL-Channel1
        self.pwconv.weight.data[2, 0, 0, 0] = -_max_weight  # MAXPOOL-Channel2
        self.pwconv.weight.data[3, 0, 0, 0] = _max_weight # AVGPOOL