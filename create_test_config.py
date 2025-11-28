
import sys

def write_config():
    # Num frames
    print("1")
    # Output path
    print("test_output_%d.png")
    # Width Height FOV
    print("200 100 90")
    
    # Camera path (static)
    # rc0 zc0 phic0
    print("15.0 4.5 3.14159")
    # Arc Azc
    print("0.0 0.0")
    # wrc wzc wc
    print("0.0 0.0 0.0")
    # prc pzc
    print("0.0 0.0")
    
    # LookAt path (static)
    # rn0 zn0 phin0
    print("0.0 4.5 0.0")
    # Arn Azn
    print("0.0 0.0")
    # wrn wzn wn
    print("0.0 0.0 0.0")
    # prn pzn
    print("0.0 0.0")
    
    # Bodies (3 bodies as per main.cu default)
    # Body 1
    print("0.0 0.0 3.0") # Center
    print("0.3 0.0 0.0") # Color
    print("3.0") # Radius
    print("1.5 0.1") # Refl Trans
    print("3") # Lights on edge
    
    # Body 2
    print("4.0 0.0 6.0")
    print("0.0 0.3 0.0")
    print("3.0")
    print("1.2 0.1")
    print("2")
    
    # Body 3
    print("8.0 0.0 9.0")
    print("0.0 0.0 0.3")
    print("3.0")
    print("1.0 0.1")
    print("1")
    
    # Floor corners
    print("-15.0 -15.0 -1.0")
    print("-15.0 15.0 -1.0")
    print("15.0 15.0 -1.0")
    print("15.0 -15.0 -1.0")
    
    # Floor texture (empty for now or existing file)
    print("../floor2.jpg") 
    # Floor tint
    print("1.0 1.0 1.0")
    # Floor reflection
    print("0.3")
    
    # Lights (4 lights)
    print("4")
    print("-15.0 -15.0 10.0") # Pos
    print("10.0 10.0 10.0") # Color
    print("-15.0 15.0 10.0")
    print("10.0 10.0 10.0")
    print("15.0 15.0 10.0")
    print("10.0 10.0 10.0")
    print("15.0 -15.0 10.0")
    print("10.0 10.0 10.0")
    
    # Render params: max_depth sqrt_rays_per_pixel
    print("5 2") # 5 depth, 4 spp

if __name__ == "__main__":
    write_config()
