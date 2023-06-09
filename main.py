import torch
from model import Discriminator, Generator, initialize_weights, saveimage, showimage

if __name__ == "__main__":
    print("Loading Session...")

    data = torch.load("data.pth")

    z_dim = data["z_dim"]
    channels_img = data["channels_img"]
    features_gen = data["features_gen"]
    features_disc = data["features_disc"]
    transform = data["transform"]
    #criterion = nn.BCELoss()
    size = int(input("How many pictures do you want to generate? : "))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(z_dim, channels_img, features_gen).to(device)
    #disc = Discriminator(channels_img, features_disc).to(device)
    initialize_weights(gen)
    #initialize_weights(disc)

    gen.eval()
    #disc.eval()
    #opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    #opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    gen.load_state_dict(data["gen"])
    #disc.load_state_dict(data["disc"])
    #opt_gen.load_state_dict(data["opt_gen"])
    #opt_disc.load_state_dict(data["opt_disc"])




    noise = torch.randn(size, z_dim, 1, 1).to(device)
    fake = gen(noise)
    saveimage(fake, "output/generated/", True)


"""
while True:
    gen.load_state_dict(data["gen"])
    disc.load_state_dict(data["disc"])
    opt_gen.load_state_dict(data["opt_gen"])
    opt_disc.load_state_dict(data["opt_disc"])

    print("getting_filepath")
    filepath = filedialog.askdirectory(initialdir=os.getcwd(), title="Select a Folder Containing Your Image")
    print(filepath)
    if filepath == "":
        break

    dataset = datasets.ImageFolder(filepath, transform)
    loader = DataLoader(dataset)

    for _, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(real.shape[0], z_dim, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()




    noise = torch.randn(1, z_dim, 1, 1).to(device)
    fake = gen(noise)
    saveimage(fake, "output/generated/")
    showimage(fake[0])
"""








print("Session Ended")