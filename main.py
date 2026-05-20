import src.io as jio
import src.calc as C
from src.cryoemsurvey import CryoEMSurvey
from src.maximamodel import MaximaModel

# USER DEFINED VARIABLES SECTION


#=============================================

#=============================================
map = jio.load_mrc_map("./data/refine_job120_postcleaned_ctf.mrc")

survey = CryoEMSurvey(
    cryo_em_map=map,
    angspix=3.026,
    survey_radius_min_angst=600,
    survey_radius_angst=900,
    grid_theta_min_deg=5,
    grid_theta_max_deg=170,
    grid_phi_min_deg=103,
    grid_phi_max_deg=175,
)

survey.calculate_direction_vectors_angst(angle_step_deg=0.2)
survey.calculate_raw_density_data()


model = MaximaModel(survey=survey,peak_filter=0.01)
model.initial_k_clustering(amount_of_clusters=4)
model.static_plot(color_by="cluster")
# model.select_cluster()
model.voxel_connection
