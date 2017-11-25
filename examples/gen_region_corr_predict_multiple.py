import subprocess

# Open the my_multiple_targets.txt
config_dir = "../config"
targets_file = "%s/%s" % (config_dir, 'my_multiple_targets.txt')
target_file = "%s/%s" % (config_dir, 'my_target.txt')
order_file = "%s/%s" % (config_dir, 'my_order.txt')

matlab_script = 'run_for_rgbdseg';

launch_generate_region_correspondence = "matlab -nodesktop -nosplash -r 'cd ../region_correspondence;multipleTargets=%d;currentTarget=%d;%s;exit;'"

launch_std2p = 'python predict_rgbdseg.py -g 0 -m ../STD2P_data/examples/models/fcn-16s-rgbd-nyud2.caffemodel';

frameStep = 3;
frameWindow = 50;

testing = False; # If true just messages will be displayed

with open(targets_file) as targets: 
   line = targets.readline()
   cnt = 1
   # Loop through target frames
   while line:
       #print("Line {}: {}".format(cnt, line.strip()))
       target = line.strip()
       # Calculate the frame range. Left limit min is 1.
       targetNum = int(target)
       leftLimit = max(1, targetNum - (frameStep * frameWindow))
       rightLimit = targetNum + (frameStep * frameWindow)
       strOrder = "%s\n%s\n" % (leftLimit, rightLimit)
       if (cnt == 1):
         bashCommand1 = "printf '%s\n' > %s" % (target, target_file)
         bashCommand2 = "printf '%s' > %s" % (strOrder, order_file)
       else:
         # Append
         bashCommand1 = "printf '%s\n' >> %s" % (target, target_file)
         bashCommand2 = "printf '%s' >> %s" % (strOrder, order_file)

       if testing:
           print(bashCommand1)
           print(bashCommand2)
       else:
           # Append the my_target.txt file with the current target frame
           subprocess.call(bashCommand1, shell=True)
           # Append the my_order.txt file with the range
           subprocess.call(bashCommand2, shell=True)

       line = targets.readline()
       cnt += 1

with open(targets_file) as targets:
   line = targets.readline()
   cnt2 = 1
   # Loop through target frames
   while line:
       #print("Line {}: {}".format(cnt, line.strip()))
       target = line.strip()
       # Calculate the frame range. Left limit min is 1.
       targetNum = int(target)

       generateRegionCorrespondence = launch_generate_region_correspondence % (cnt-1, cnt2, matlab_script)

       if testing:
           print(generateRegionCorrespondence)
           print(launch_std2p)
       else:
           a = 1 + 1 
           # Generate the region correspondences
           subprocess.call(generateRegionCorrespondence, shell=True)
           # Launch the predict_rgbdseg.py script
           #subprocess.call(launch_std2p, shell=True)

       line = targets.readline()
       cnt2 += 1
       exit()
      
