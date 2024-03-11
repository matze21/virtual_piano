Different computer vision techniques to detect hands in an image, track the finger tips and detect when they hit a surface in order to progam an artificial piano you could play only with a laptop camera on a table top or other surfaces


Problems:
- default CNN is not well trained for the hand positions similar to pianos -> noisy data
- contact estimation is hardly distinguishable from tracked signal of CNN finger tips (tried optical flow, time tracked hand position, camera based depth estimation)
- surface is only known when using a camera calibration in advance, parameters can't be read out of system or approximated through hand position (pseudo-inverse is not good enough for underdetermined linear system)

TODO:
- improve CNN by different hand positions
- work on automatic calibration

Note:
this is a prototype and code quality is no priority as of now

  
