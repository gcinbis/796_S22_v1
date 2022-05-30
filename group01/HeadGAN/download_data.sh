unzip TDDFA_V2.zip
mkdir data
mkdir data/test/
mkdir data/test/aac
mkdir data/test/mp4
wget --user USERNAME --password PASSWORD -P data/test/aac http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_aac.zip
wget --user USERNAME --password PASSWORD -P data/test/mp4 http://cnode01.mm.kaist.ac.kr/voxceleb/vox1a/vox2_test_mp4.zip
cd data/test/aac
unzip vox2_test_aac.zip
rm vox2_test_aac.zip
cd ../mp4
unzip vox2_test_mp4.zip
rm vox2_test_mp4.zip
