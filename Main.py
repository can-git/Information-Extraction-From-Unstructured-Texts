import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import Properties as p
from BertClassifier import BertClassifier
from TextDataset import TextDataset
from Preprocess import Preprocess


def train(model, train_data, val_data, criterion, optimizer):

    train, val = TextDataset(train_data, True), TextDataset(val_data, True)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=p.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=p.BATCH_SIZE, pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = nn.DataParallel(model, device_ids=[0])
        model.to(device)
        model.train()

    for epoch_num in range(p.EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for _, train_input, train_label in train_dataloader:
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(torch.log(output), train_label.argmax(dim=1))
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label.argmax(dim=1)).sum().item() / len(output.argmax(dim=1))
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for _, val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(torch.log(output), val_label.argmax(dim=1))
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label.argmax(dim=1)).sum().item() / len(output.argmax(dim=1))
                total_acc_val += acc

        # print(
        #     f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .5f} \
        #         | Train Accuracy: {total_acc_train / len(train_dataloader): .5f}')
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .5f} \
                        | Train Accuracy: {total_acc_train / len(train_dataloader): .5f} \
                        | Val Loss: {total_loss_val / len(val_dataloader): .5f} \
                        | Val Accuracy: {total_acc_val / len(val_dataloader): .5f}')
    if p.SAVE_MODEL:
        torch.save(model.state_dict(), 'model.pt')


def evaluate(model, test_data):
    test = TextDataset(test_data, False)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    predicted_values = []
    predicted_ids = []
    with torch.no_grad():

        for test_ids, test_input, _ in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            predicted_ids.append("".join(test_ids))
            predicted_values.append(output.cpu().detach().numpy())

    df = pd.concat([pd.DataFrame(i) for i in predicted_values], ignore_index=True)

    new_df = pd.DataFrame({"pateint_id": predicted_ids,
                           "likelihood_G1": df.iloc[:, 0],
                           "likelihood_G2": df.iloc[:, 1],
                           "likelihood_G3": df.iloc[:, 2],
                           "likelihood_G4": df.iloc[:, 3]})
    new_df.to_csv("can_yilmaz_assignment_3.csv", index=False)


df_train, df_val, df_test = Preprocess().getItem()

model = BertClassifier()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=p.LR)
train(model, df_train, df_val, criterion, optimizer)

# model.load_state_dict(torch.load('model.pt'))  # Uncommenting for trying model to get csv
# evaluate(model, df_test)

