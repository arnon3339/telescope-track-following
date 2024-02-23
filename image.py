from PIL import Image, ImageDraw
import numpy as np

IMG_PATH = "/home/arnon/Projects/telescope-track-following/imgs/"

def gen_4hitmaps(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.05), int(hiegth*2.25)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2), int(hiegth*(int(i/2))) + 250)
        )
    # im.show()
    im.save(IMG_PATH + "hit4maps.png")

def gen_2hitmaps(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.05), int(hiegth*1.05)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2), int(hiegth*(int(i/2))) + 100)
        )
    # im.show()
    im.save(IMG_PATH + "hit2maps.png")

def gen_2tracks(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, height = sam_im.size
    im = Image.new('RGB', [int(width*1.8), int(height*0.7)],
                   "#ffffff")
    left = width/10
    top = height / 4
    right = width
    bottom = height
    for i in range(len(file_list)):
        im_im = Image.open(IMG_PATH + file_list[i])
        im_im = im_im.crop((left, top, right, bottom))
        im.paste(im_im,
            box=(int(width*i) - 150*i, 0)
        )
    # im.show()
    im.save(IMG_PATH + "track2en.png")

def gen_2tracks_pix(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, height = sam_im.size
    im = Image.new('RGB', [int(width*1.8), int(height*0.7)],
                   "#ffffff")
    left = width/10
    top = height / 4
    right = width
    bottom = height
    for i in range(len(file_list)):
        im_im = Image.open(IMG_PATH + file_list[i])
        im_im = im_im.crop((left, top, right, bottom))
        im.paste(im_im,
            box=(int(width*i) - 150*i, 0)
        )
    # im.show()
    im.save(IMG_PATH + "track2en_pix.png")

def gen_4cluter_size(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.10), int(hiegth*2.25)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2) - (int(i/2) - 1)*80, int(hiegth*(int(i/2))) + 250)
        )
    im.save(IMG_PATH + "clutersize4.png")

def gen_2effexps(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.05), int(hiegth*1.05)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2), int(hiegth*(int(i/2))) + 100)
        )
    # im.show()
    im.save(IMG_PATH + "eff2smax.png")

def gen_2eventhit(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.05), int(hiegth*1.05)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2), int(hiegth*(int(i/2))) + 100)
        )
    # im.show()
    im.save(IMG_PATH + "eventhits.png")

def gen_6hitmaps(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*3.05), int(hiegth*2.25)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%3)) + 80*(i%3), int(hiegth*(int(i/3))) + 250)
        )
    # im.show()
    im.save(IMG_PATH + "hit6maps.png")

def gen_2clusterhist(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2.05), int(hiegth*1.05)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)) + 100*(i%2), int(hiegth*(int(i/2))) + 100)
        )
    # im.show()
    im.save(IMG_PATH + "clusterthits.png")

def gen_2hit3D(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*2), int(hiegth*1)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%2)), int(hiegth*(int(i/2))))
        )
    # im.show()
    im.save(IMG_PATH + "hit3D.png")

def gen_9clusters(file_list):
    sam_im = Image.open(IMG_PATH + file_list[0])
    width, hiegth = sam_im.size
    im = Image.new('RGB', [int(width*3.05), int(hiegth*3.25)],
                   "#ffffff")
    hiegth = hiegth*1.05
    for i in range(len(file_list)):
        im.paste(Image.open(
            IMG_PATH + file_list[i]   
            ),
            box=(int(width*(i%3)), int(hiegth*(int(i/3))))
        )
    # im.show()
    im.save(IMG_PATH + "cluster9samples.png")
# gen_2hitmaps([
#     "Col70MeV1000MU.png",
#     "Col200MeV500MU.png"
# ])

# gen_4hitmaps(
#     ["MultipleBeam70MeV100MU.png", "MultipleBeam70MeV100MU.png",
#     "MultipleAcrylic70MeV1000MU.png", "MultipleAcrylic200MeV500MU.png"]
# )

# gen_2tracks([
#     "70_MeV_Experiment_rec_tracks.png",
#     "200_MeV_Experiment_rec_tracks.png"
# ])
# gen_2tracks_pix([
#     "70_MeV_Experiment_pixel_rec_tracks.png",
#     "200_MeV_Experiment_pixel_rec_tracks.png"
# ])

# gen_4cluter_size([
#     "sim_acrylic_70MeV.png",
#     "Cluster_size_acrylic_70MeV.png",
#     "sim_acrylic_200MeV.png",
#     "Cluster_size_acrylic_200MeV.png",
# ])

# gen_2eventhit([
#     "ncluster_event_total.png",
#     "ncluster_event_filtered.png"
# ])

# gen_6hitmaps([
#     "col_beam_70MeV.png",
#     "col_beam_100MeV.png",
#     "col_beam_120MeV.png",
#     "col_beam_150MeV.png",
#     "col_beam_180MeV.png",
#     "col_beam_200MeV.png"
# ])

gen_6hitmaps([f"col_beam_70MeV_layer{i}.png" for i in range(6)])

# gen_2clusterhist([
#    "sigma_cluster_2sigma_070MeV.png" ,
#    "sigma_cluster_2sigma_200MeV.png"
# ])

# gen_2hit3D([
#     "simhit_70MeV.png",
#     "simhit_200MeV.png"
# ])

# gen_9clusters([f"Figure_{i + 1}.png" for i in range(9)])