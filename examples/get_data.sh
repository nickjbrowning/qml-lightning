MD17_files_out=(benzene_dft.npz uracil_dft.npz naphthalene_dft.npz salicylic_dft.npz malonaldehyde_dft.npz ethanol_dft.npz toluene_dft.npz aspirin_dft.npz)
MD17_files_in=(benzene2017_dft.npz uracil_dft.npz naphthalene_dft.npz salicylic_dft.npz malonaldehyde_dft.npz ethanol_dft.npz toluene_dft.npz aspirin_dft.npz)

mkdir -p data

echo "---Downloading MD17---"

for index in "${!MD17_files_in[@]}"; 
do
	echo "${MD17_files_out[$index]}"
	if [[ ! -f data/"${MD17_files_out[$index]}" ]]; then
		wget "http://www.quantum-machine.org/gdml/data/npz/${MD17_files_in[$index]}" -O data/"${MD17_files_out[$index]}"
	fi
done

echo "---Downloading QM9---"

if [[ ! -f "data/qm9_data.npz" ]]; then
	wget https://ndownloader.figshare.com/files/28893843 -O "data/qm9_data.npz"
fi

echo "---ACS doesn't like wget - download the 3BPA zipfile from the paper ---"

xdg-open  https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.1c00647/

if [[ ! -d data/3BPA ]]; then
  mkdir data/3BPA
fi



	
	