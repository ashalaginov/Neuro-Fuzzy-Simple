## -*- Makefile -*-
##
## User: andriis
## Time: Feb 5, 2018 1:22:58 PM
## Makefile created by Oracle Solaris Studio.
##
## This file is generated automatically.
##


#### Compiler and tool definitions shared by all build targets #####
CCC = g++
CXX = g++
BASICOPTS = -m64 -fopenmp
CCFLAGS = $(BASICOPTS)
CXXFLAGS = $(BASICOPTS)
CCADMIN = 


# Define the target directories.
TARGETDIR_neuro-fuzzy-simple=GNU-amd64-Linux


all: $(TARGETDIR_neuro-fuzzy-simple)/neuro-fuzzy-simple

## Target: neuro-fuzzy-simple
OBJS_neuro-fuzzy-simple =  \
	$(TARGETDIR_neuro-fuzzy-simple)/main.o
USERLIBS_neuro-fuzzy-simple =  $(SYSLIBS_neuro-fuzzy-simple) 
DEPLIBS_neuro-fuzzy-simple =  
LDLIBS_neuro-fuzzy-simple = $(USERLIBS_neuro-fuzzy-simple)


# Link or archive
$(TARGETDIR_neuro-fuzzy-simple)/neuro-fuzzy-simple: $(TARGETDIR_neuro-fuzzy-simple) $(OBJS_neuro-fuzzy-simple) $(DEPLIBS_neuro-fuzzy-simple)
	$(LINK.cc) $(CCFLAGS_neuro-fuzzy-simple) $(CPPFLAGS_neuro-fuzzy-simple) -o $@ $(OBJS_neuro-fuzzy-simple) $(LDLIBS_neuro-fuzzy-simple)


# Compile source files into .o files
$(TARGETDIR_neuro-fuzzy-simple)/main.o: $(TARGETDIR_neuro-fuzzy-simple) main.cpp
	$(COMPILE.cc) $(CCFLAGS_neuro-fuzzy-simple) $(CPPFLAGS_neuro-fuzzy-simple) -o $@ main.cpp 

        

#### Clean target deletes all generated files ####
clean:
	rm -f \
		$(TARGETDIR_neuro-fuzzy-simple)/neuro-fuzzy-simple \
		$(TARGETDIR_neuro-fuzzy-simple)/main.o
	$(CCADMIN)
	rm -f -r $(TARGETDIR_neuro-fuzzy-simple)


# Create the target directory (if needed)
$(TARGETDIR_neuro-fuzzy-simple):
	mkdir -p $(TARGETDIR_neuro-fuzzy-simple)


# Enable dependency checking
.KEEP_STATE:
.KEEP_STATE_FILE:.make.state.GNU-amd64-Linux

