import json, os

zipFile_location = "~/Project/source/coupled_flux_qubit_protocol/myarch_mothra.zip"

os.system(f"scp tkwtang@mothra.csc.ucdavis.edu:~/source/coupled_flux_qubit_protocol/myarch_mothra.zip {zipFile_location}")

os.system(f"unzip -o {zipFile_location} -d ~/Project/source/coupled_flux_qubit_protocol/myarch_mothra")

galleryPath_local = "/Users/edwardtang/Project/source/coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery"
galleryPath_mothra = "/Users/edwardtang/Project/source/coupled_flux_qubit_protocol/myarch_mothra"
fileArray = os.listdir(galleryPath_mothra)

with open(os.path.join(galleryPath_local, "gallery.json")) as f:
    TOC_local = json.load(f)
with open(os.path.join(galleryPath_mothra, "gallery_mothra.json")) as f:
    TOC_mothra = json.load(f)

simulationIDArray_local = []
simulationIDArray_mothra = []

for i, x in enumerate(TOC_local):
    simulationIDArray_local.append([i, x["simulation_data"]["simulation_id"]])

for i, x in enumerate(TOC_mothra):
    simulationIDArray_mothra.append([i, x["simulation_data"]["simulation_id"]])

for x in simulationIDArray_mothra:
    TOC_local.append(TOC_mothra[x[0]]) # x[0] is the index, TOC_mothra is the json file
    extension = ["_szilard_engine.mp4", "_work_statistic.png", "_work_distribution.png", "_work_statistic.npy", "_work_distribution.npy"]
    for ext in extension:
        old_location = os.path.join(galleryPath_mothra, x[1] + ext)
        new_location = os.path.join(galleryPath_local, x[1] + ext)
        try:
            os.rename(old_location, new_location)
        except:
            print("extension does not exist")

with open(os.path.join(galleryPath_local, "gallery.json"), "w") as f:
    json.dump(TOC_local, f)
