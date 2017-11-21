# Open the my_multiple_targets.txt
config_dir = "../config"
targets_file = "%s/%s" % (config_dir, 'my_multiple_targets.txt')
target_file = "%s/%s" % (config_dir, 'my_target.txt')

launch_std2p = 'python predict_rgbdseg.py -g 0 -m ../STD2P_data/examples/models/fcn-16s-rgbd-nyud2.caffemodel';
  
with open(targets_file) as targets: 
   line = targets.readline()
   cnt = 1
   # Loop through target frames
   while line:
       #print("Line {}: {}".format(cnt, line.strip()))
       target = line.strip()
       bashCommand = "echo %s > %s" % (target, target_file)
       import subprocess
       # Replace the my_target.txt file with the current target frame
       subprocess.call(bashCommand, shell=True)
       # Launch the predict_rgbdseg.py script
       subprocess.call(launch_std2p, shell=True)

       line = targets.readline()
       cnt += 1

# I tried to make this as a matlab script at first but it didn't work when calling python because python libraries couldn't be loaded correctly
# For each target frame launch predict_rgbdseg.py replacing my_target.txt every time with the new target frame
#for ii = 1 : length(targets)
#  target = targets(ii);
#  command = sprintf('echo %d > %s', target, target_file);
#  system(command);
#  command = sprintf('%s; %s', setup_env_params, launch_std2p)
#  system(launch_std2p);
#end 
