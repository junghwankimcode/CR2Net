import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import svd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
trainSNRVec = np.arange(-10, 31, 5)
TrainSet = int(1e4)
TestSet = int(1e4)
trainDoF = 8
P = 16
Kfactor = 3
subcarrier_bandwidth = 60e3
subcarrier_spacing = 15
excess_delay = np.random.uniform(0, 1e-6)
batch_size = 128
epochs = 1000

# Functions
def make_hankel(x):
    l = len(x)
    r = int(np.ceil(l / 2))
    c = l - r + 1
    m = np.zeros((r, c), dtype=np.complex64)
    for i in range(r):
        m[i, :] = x[i:i + c]
    _, s, _ = svd(m, full_matrices=False)
    return s.real  # No padding — dynamic length

def generate_signal(snr_db, rician):
    IT = np.random.randn(trainDoF) + 1j * np.random.randn(trainDoF)
    CR = np.exp(-1j * 2 * np.pi * excess_delay * subcarrier_bandwidth * subcarrier_spacing * np.random.rand(trainDoF))
    base = np.zeros(P, dtype=np.complex64)
    for d in range(trainDoF):
        base += IT[d] * CR[d] ** np.arange(P)
    xx = base / np.linalg.norm(base)
    if rician:
        IT_d = np.random.randn() + 1j * np.random.randn()
        CR_d = np.exp(1j * 2 * np.pi * np.random.rand())
        x_dominant = IT_d * CR_d ** np.arange(P)
        x_dominant *= np.sqrt(10 ** (Kfactor / 10))
        xx = xx + x_dominant
        xx = xx / np.linalg.norm(xx)
    signal_power = np.mean(np.abs(xx) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(P) + 1j * np.random.randn(P))
    return xx + noise

# Neural net with dynamic input dimension
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).squeeze(-1)

# Main
detectionRate1 = []
detectionRate2 = []
detectionRate3 = []
detectionRate4 = []

for snr in trainSNRVec:
    trainData1 = []
    trainData2 = []
    trainData3_Rician = []
    trainData3_Rayleigh = []
    trainAns = []

    for _ in range(TrainSet):
        label = np.random.randint(2)
        xn = generate_signal(snr, rician=(label == 1))
        trainData1.append(np.abs(xn))
        hankel_s = make_hankel(xn)
        trainData2.append(hankel_s)
        trainAns.append(float(label))
        spectrum = np.abs(np.fft.ifft(xn))
        if label == 1:
            trainData3_Rician.append(spectrum)
        else:
            trainData3_Rayleigh.append(spectrum)

    # Skewness/kurtosis thresholds
    skew_r = np.mean(skew(trainData3_Rician, axis=1))
    skew_y = np.mean(skew(trainData3_Rayleigh, axis=1))
    kurt_r = np.mean(kurtosis(trainData3_Rician, axis=1))
    kurt_y = np.mean(kurtosis(trainData3_Rayleigh, axis=1))
    skew_thresh = 0.5 * (skew_r + skew_y)
    kurt_thresh = 0.5 * (kurt_r + kurt_y)

    # Convert to tensor and move to device
    X1 = torch.tensor(trainData1, dtype=torch.float32).to(device)
    X2 = torch.tensor(trainData2, dtype=torch.float32).to(device)
    Y = torch.tensor(trainAns, dtype=torch.float32).to(device)

    model1 = Net(X1.shape[1]).to(device)
    model2 = Net(X2.shape[1]).to(device)
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader1 = DataLoader(TensorDataset(X1, Y), batch_size=batch_size, shuffle=True)
    loader2 = DataLoader(TensorDataset(X2, Y), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader1:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model1(xb)
            loss = loss_fn(pred, yb)
            opt1.zero_grad(); loss.backward(); opt1.step()
        for xb, yb in loader2:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model2(xb)
            loss = loss_fn(pred, yb)
            opt2.zero_grad(); loss.backward(); opt2.step()

    # Test phase
    testData1 = []
    testData2 = []
    testData3 = []
    testAns = []

    for _ in range(TestSet):
        label = np.random.randint(2)
        xn = generate_signal(snr, rician=(label == 1))
        testData1.append(np.abs(xn))
        testData2.append(make_hankel(xn))
        testData3.append(np.abs(np.fft.ifft(xn)))
        testAns.append(label)

    testData1 = torch.tensor(testData1, dtype=torch.float32).to(device)
    testData2 = torch.tensor(testData2, dtype=torch.float32).to(device)
    testAns = np.array(testAns)

    with torch.no_grad():
        pred1 = (model1(testData1).cpu().numpy() > 0.5).astype(int)
        pred2 = (model2(testData2).cpu().numpy() > 0.5).astype(int)

    pred3 = (kurtosis(testData3, axis=1) > kurt_thresh).astype(int)
    pred4 = (skew(testData3, axis=1) > skew_thresh).astype(int)

    acc1 = np.mean(pred1 == testAns)
    acc2 = np.mean(pred2 == testAns)
    acc3 = np.mean(pred3 == testAns)
    acc4 = np.mean(pred4 == testAns)

    detectionRate1.append(acc1)
    detectionRate2.append(acc2)
    detectionRate3.append(acc3)
    detectionRate4.append(acc4)

# Plot
plt.plot(trainSNRVec, detectionRate2, label='CR2Net')
plt.plot(trainSNRVec, detectionRate1, label='Deep learning-based (legacy)')
plt.plot(trainSNRVec, detectionRate4, label='Skewness-based')
plt.plot(trainSNRVec, detectionRate3, label='Kurtosis-based')
plt.xlabel('SNR [dB]')
plt.ylabel('Detection rate')
plt.legend(loc='best')
plt.grid(True)
plt.ylim(0.3, 1.0)
plt.xticks(trainSNRVec)
plt.show()
