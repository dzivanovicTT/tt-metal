# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=8278e02931ec31bafe9ab267480c3c16
sfpi_x86_64_Linux_deb_md5=3d564442a71f92a365ce00ab249d635d
sfpi_x86_64_Linux_rpm_md5=397146785b990e21176edb187ef39dd3
