# How To Run 12g-LowLevelController-TransformedDynamics

## Dependencies

- DART (at least version 6.3)
 [Dart Homepage](https://dartsim.github.io)
- config4cpp
 [Source Code Download](http://config4star.org/#main-source-code)
  
 How to add config4cpp to your system (Linux/BSD/OSX)
 
  1: Download and extract the source code of config4cpp
  
  2: cd config4cpp
  
  3: Follow the README.txt in config4cpp/
  
  4: Run the following commands to add the make'd files into your local system
		
    sudo cp bin/{config2cpp,config4cpp} /usr/local/bin &&
    sudo cp lib/libconfig4cpp.a /usr/local/lib &&
    sudo cp -r include/config4cpp /usr/local/include &&
    sudo chmod g+rx /usr/local/include/config4cpp &&
    sudo chmod o+rx /usr/local/include/config4cpp
    
  *Note: You can just copy paste the above block of commands

## Download this Repository
 - Clone this repo
 or
 - Download this repo and extract from the zip file

## Clean up Downloaded Files

 - Remove the build folder in this project

## Build and Run the Project

 1: Enter the cloned/downloaded project directory
 
 2: Build the Project
 
	mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    
 3: Run the Project

    ./LowLevelController
   Press [Space] to start Dart simulation from pop-up window
