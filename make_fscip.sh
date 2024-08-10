path_to_scip="<set path here>, e.g. /home/user/Downloads/scipoptsuite-7.0.2"
path_to_boost="<set path here>, e.g. /opt/boost_1_76_0"

cd $path_to_scip
# in scip dir (scipoptsuite-7.0.2):
# NOTE: it did not work with ZIMPL=true in the first place to build scip, however after the overall build failed, the ZIML libraries have been built. Then the linker flags for the zimpl library have been added and the overal build succeeded:
make PARASCIP=true ZLIB=true ZIMPL=true CXXFLAGS+=-I$path_to_boost LDFLAGS+=-L$path_to_boost/stage/lib LDFLAGS+=-lboost_program_options LDFLAGS+=-lgmp LDFLAGS+=-lreadline LDFLAGS+=-lz LDFLAGS+=-lm
make PARASCIP=true ZLIB=true ZIMPL=true CXXFLAGS+=-I$path_to_boost LDFLAGS+=-L$path_to_boost/stage/lib LDFLAGS+=-lboost_program_options LDFLAGS+=-lgmp LDFLAGS+=-lreadline LDFLAGS+=-lz LDFLAGS+=-lm LDFLAGS+=-L$path_to_scip/zimpl/lib LDFLAGS+=-lzimpl.linux.x86_64.gnu.opt

# make ug (all the linker flags were not required when scip was built with ZLIB=false ZIMPL=false):
cd ug/lib
ln -s ../../scip scip
cd ..
make LDFLAGS+=-L$path_to_scip/zimpl/lib LDFLAGS+=-lzimpl.linux.x86_64.gnu.opt LDFLAGS+=-lgmp LDFLAGS+=-lreadline LDFLAGS+=-lz LDFLAGS+=-lpthread
