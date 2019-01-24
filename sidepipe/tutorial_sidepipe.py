from spipeline import SPipeline, SStream
import yaml

in_file_name = "./taaa.wav"
out_file_name = "feat"
sp = SPipeline("tutorial.yaml")
sp.connect_io([in_file_name], [out_file_name])
sp.run()