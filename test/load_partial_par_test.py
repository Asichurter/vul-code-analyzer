import torch

class TestModel_1(torch.nn.Module):
    def __init__(self):
        super(TestModel_1, self).__init__()
        self.module_1 = torch.nn.Linear(2, 2)
        self.weight_1 = torch.nn.Parameter(torch.randn((2,2)))

    def forward(self, input_data):
        return self.module_1(torch.mm(self.weight_1, input_data))

class TestModel_2(torch.nn.Module):
    def __init__(self):
        super(TestModel_2, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn((2,2)))

    def forward(self, input_data):
        return torch.mm(self.weight_1, input_data)


model1 = TestModel_1()
model2 = TestModel_2()
state_dict = model2.state_dict()

new_state_dict = {
    'module_1.weight': state_dict['weight']
}


