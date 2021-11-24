class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #베르누이 함수에 넣기 위해 input이 0~1사이의 값이 되어야 한다.
        x = F.sigmoid(self.fc3(x))
        return x

'''State of CartPole
카트의 위치
카트의 속력
막대기의 각도
막대기의 끝부분(상단) 속도
'''
