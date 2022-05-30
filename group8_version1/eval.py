# implement evaluation functions
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from data import AnimeFacesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import histogram_feature_v2
from loss import hellinger_dist_loss

def random_interpolate_hists(batch_data, device="cpu"):
    B = batch_data.size(0)
    delta = torch.rand((B,1,1,1)).to(device)
    first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    first_hist = histogram_feature_v2(first_images, device=device)
    second_hist = histogram_feature_v2(second_images, device=device)
    hist_t = delta*first_hist + (1-delta)*second_hist
    return hist_t

def fake_generator(batch_size):
    return torch.randint(0, 255, (batch_size, 3, 256, 256))

# Device "cpu" is advised by torch metrics
def fid_scores(generator, test_path, fid_batch=16, device="cpu"):
    transform = transforms.Compose([transforms.Resize((256,256))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=fid_batch, shuffle=True)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True)
    
    fids = []
    num_generated = 0
    for batch_data in dataloader:
        z = torch.rand(batch_data.size(0), 512).to(device)
        target_hist = random_interpolate_hists(batch_data)
        fake_data = fake_generator(fid_batch)
        batch_data = batch_data.byte()  # Convert to uint8 for fid
        fake_data = fake_data.byte()
        #  fake_data = generator(z, target_hist)
        fid.update(batch_data, real=True)
        fid.update(fake_data, real=False)
        batch_fid = fid.compute()
        fids.append(batch_fid.item())
        num_generated += fid_batch
        if num_generated > 10000: break

    return fids

def hist_uv_kl(generator, test_path, kl_batch=64, device="cpu"):
    transform = transforms.Compose([transforms.Resize((256,256))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=kl_batch, shuffle=True)
    
    kls = []
    num_generated = 0
    for batch_data in dataloader:
        z = torch.rand(batch_data.size(0), 512).to(device)
        target_hist = random_interpolate_hists(batch_data)
        fake_data = fake_generator(kl_batch)
        #  fake_data = generator(z, target_hist)
        fake_hist = histogram_feature_v2(fake_data, device=device)
        print(torch.mean(fake_hist), torch.norm(fake_hist), torch.max(fake_hist), torch.min(fake_hist))
        kl = torch.nn.functional.kl_div(fake_hist, target_hist)
        kls.append(kl)       
        print(kl) 
        num_generated += kl_batch
        if num_generated > 10000: break
    
    return kls

def hist_uv_h(generator, test_path, h_batch=64, device="cpu"):
    transform = transforms.Compose([transforms.Resize((256,256))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=h_batch, shuffle=True)
    
    hs = []
    num_generated = 0
    for batch_data in dataloader:
        z = torch.rand(batch_data.size(0), 512).to(device)
        # target_hist = random_interpolate_hists(batch_data)
        target_hist = histogram_feature_v2(batch_data, device=device)
        fake_data = fake_generator(h_batch)
        #  fake_data = generator(z, target_hist)
        h = hellinger_dist_loss(fake_data, target_hist, device=device)
        hs.append(h)       
        print(h) 
        num_generated += h_batch
        if num_generated > 10000: break
    
    return hs

def main():
    # generate two slightly overlapping image intensity distributions
    # fid_scores(None, "images", fid_batch=8)
    hist_uv_h(None, "images")
if __name__=="__main__": main()