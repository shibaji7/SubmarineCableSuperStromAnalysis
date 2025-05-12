import datetime as dt
from loguru import logger
import matplotlib.dates as mdates

from scubas.cables import TransmissionLine, Cable
import sys
sys.path.append("py/")

from utils import StackPlots,  get_cable_informations, clean_B_fields

class SCUBASModel(object):
    
    def __init__(
            self, 
            cable_name="TAT-8",
            cable_structure=get_cable_informations(),
            segment_files=[],
        ):
        self.cable_name = cable_name
        self.cable_structure = cable_structure
        self.segment_files = segment_files
        logger.info(f"Initialize {cable_name}")
        return
    
    def read_stations(
            self, 
            stns=["FRD", "STJ", "HAD"], 
            stn_files=[
                ["dataset/May2024/frd20240510psec.sec.txt"],
                ["dataset/May2024/had20240510psec.sec.txt"], 
                ["dataset/May2024/stj20240510psec.sec.txt"]
            ]
        ):
        self.frames = clean_B_fields(stns, stn_files)
        return
    
    def initialize_TL(self):
        self.tlines = []
        for i, seg in enumerate(self.cable_structure.cable_seg):
            self.tlines.append(
                TransmissionLine(
                    sec_id=seg["sec_id"],
                    directed_length=dict(
                        edge_locations=dict(
                            initial=seg["initial"], 
                            final=seg["final"]
                        )
                    ),
                    elec_params=dict(
                        site=seg["site"],
                        width=seg["width"],
                        flim=seg["flim"],
                    ),
                    active_termination=seg["active_termination"],
                ).compile_oml(self.segment_files[i]),
            )
        return
    
    def run_cable_segment(self):
        # Running the cable operation
        logger.info(f"Components: {self.tlines[0].components}")
        self.cable = Cable(self.tlines, self.tlines[0].components)
        return
    
    def plot_TS_with_others(
        self, fname, date_lim=[], fig_title="", vlines=[],
        major_locator=mdates.MinuteLocator(byminute=range(0, 60, 5)),
        minor_locator=mdates.MinuteLocator(byminute=range(0, 60, 1)), 
        text_size=15
    ):
        sp = StackPlots(nrows=1, ncols=1, datetime=True, figsize=(6, 4), text_size=text_size)
        _, ax = sp.plot_stack_plots(
            self.cable.tot_params.index, self.cable.tot_params["Vt(v)"], 
            ylim=[-3000, 3000], xlim=date_lim, title=fig_title,
            xlabel="Time, UT", ylabel="Cable Voltage, V"
        )        
        sp.save_fig(fname)
        sp.close()
        return