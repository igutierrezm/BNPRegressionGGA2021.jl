FROM rocker/r-ver:4.1.0

# Install default-jre git, glpk-utils, libxt6, wget, xml2, xdg-utils
RUN apt-get update 
RUN apt-get -y install \
    default-jre default_jdk \
    git glpk-utils libxt6 wget xml2 xdg-utils

# Configure Java in R
RUN R CMD javareconf 

# Copy libjvm.so to a more appropriate location
RUN sudo ln -s \
    /usr/lib/jvm/java-1.11.0-openjdk-amd64/lib/server/libjvm.so \
    /usr/local/lib/R/lib/libjvm.so

# Install some useful R packages from CRAN
RUN install2.r --error --skipinstalled --ncpus -1 \
    dplyr ggplot2 glmulti JuliaConnectoR languageserver \
    leaps LPKsample readr remotes R.rsp tidyr

# Install some useful R packages from GitHub
RUN R -e "remotes::install_github('nyiuab/BhGLM', force = TRUE)"

# Install julia 1.7.0
WORKDIR /opt/
ARG JULIA_TAR=julia-1.7.0-linux-x86_64.tar.gz
RUN wget -nv https://julialang-s3.julialang.org/bin/linux/x64/1.7/${JULIA_TAR}
RUN tar -xzf ${JULIA_TAR}
RUN rm -rf ${JULIA_TAR}
RUN ln -s /opt/julia-1.7.0/bin/julia /usr/local/bin/julia

# Add libR.so to the LD_LIBRARY_PATH environment variable
ARG libRdir=/usr/local/lib/R/lib
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH':$libRdir >> ~/.bashrc

# Update the libstdc++.so.6 used by julia
RUN cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/julia-1.7.0/lib/julia/

# To build this image, run
# sudo docker build -t dgsbp:0.1 .

# To create a container, run
# sudo docker run -t -d dgsbp:0.1
