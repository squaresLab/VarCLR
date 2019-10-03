# Code adapted from SentEval: https://github.com/facebookresearch/SentEval

data_path=.

declare -A STS17

STS17="track1.ar-ar track2.ar-en track3.es-es track4a.es-en track5.en-en track6.tr-en"

mkdir $data_path/STS

wget http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip -P $data_path/STS
wget http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip -P $data_path/STS
unzip $data_path/STS/sts2017.eval.v1.1.zip -d $data_path/STS
unzip $data_path/STS/sts2017.gs.zip -d $data_path/STS
mv $data_path/STS/STS2017.eval.v1.1 $data_path/STS/STS17-test
rm $data_path/STS/sts2017.eval.v1.1.zip $data_path/STS/sts2017.gs.zip
mv $data_path/STS/STS2017.gs/* $data_path/STS/STS17-test
rm -Rf $data_path/STS/STS2017.gs

for sts_task in ${STS17}
do
    fname=STS.input.$sts_task.txt
    task_path=$data_path/STS/STS17-test

    mv $task_path/$fname $task_path/tmp1
    paste $task_path/tmp1 $task_path/STS.gs.$sts_task.txt > $task_path/$fname
    rm $task_path/tmp1
done