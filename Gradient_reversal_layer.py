class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb=1.0):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lamb, None
		
		
		
class LinearLayer(torch.nn.Module):
	def __init__(self):
		super(LinearLayer, self).__init__()
		self.fc = torch.nn.Linear(a, b)
		
	def forward(self, x):
		x = GradReverse.apply(x)
		y = self.fc(x)
		return y