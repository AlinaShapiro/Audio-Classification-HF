import traceback, sys
import PyQt5
from PyQt5 import QtCore, QtWidgets, QtMultimedia 
from PyQt5.QtChart import QChart, QChartView, QPieSeries
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPainter, QPen, QIcon

import os 
import time
from moviepy.editor import *
import numpy as np
import torchaudio
import torch
import json

from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("models/emocall/wavlm/audio-model", 
                                                             trust_remote_code=True)
model = WavLMForSequenceClassification.from_pretrained("models/emocall/wavlm/audio-model",
                                                         trust_remote_code=True).to(device)
                                                    
num2emotion = model.config.id2label

class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

class Worker(QtCore.QRunnable):
    '''
    Worker thread
    '''
    def __init__(self, fn, str, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.str = str
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
    
    
    @QtCore.pyqtSlot()
    def run(self):
        '''
        Your code goes in this function
        '''
        print("Thread start")
        print(self.str)
        print(self.args, self.kwargs)
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QtWidgets.QWidget):

    firstDetect = True

    def __init__(self):
        super().__init__()
        self.emo_res = dict()
        self.info = "Поддерживаемые форматы файлов: аудио - *.wav *.mp3 *.m4a; видео - *.mp4"
        #Select
        self.selectFileButton = QtWidgets.QPushButton("Выберите файл")
        self.selectFileButton.clicked.connect(self.selectFile)
        
        #Selected info
        self.selectedFileText = QtWidgets.QLabel(text = self.info)  

        self.createPieChart()

        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setValue(0)
        self.pbar.hide()
        #Media player
        self.mediaPlayer = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)

        self.mediaPlayer.stateChanged.connect(self.mediastateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        #Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.setSlidePosition)
        #Play button
        self.playButton = QtWidgets.QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        #Detect button
        detectButton = QtWidgets.QPushButton("Анализировать эмоции")
        detectButton.clicked.connect(self.detect)

        self.videoWidget = QVideoWidget()
        self.videoWidget.setMaximumHeight(450)
        self.mediaPlayer.setVideoOutput(self.videoWidget) 

        self.dumpButton = QtWidgets.QPushButton("Выгрузить отчет")
        self.dumpButton.clicked.connect(self.dumpToFile)
        self.dumpButton.setDisabled(True)
        #Vertiacal Layout and adding widgets to it
        vLayout = QtWidgets.QVBoxLayout()

        vLayout.addWidget(self.selectFileButton)
        vLayout.addWidget(self.videoWidget)
        vLayout.addWidget(self.chartView)
        vLayout.addWidget(self.selectedFileText)
        vLayout.addWidget(self.pbar)
        vLayout.addWidget(self.slider)
        vLayout.addWidget(self.playButton)

        vLayout.addWidget(detectButton)
        vLayout.addWidget(self.dumpButton)
        #Main Window Geometry
        self.setGeometry(300, 300, 1020, 800)
        self.setLayout(vLayout)
        self.setWindowTitle("Анализ Эмоционального Cостояния")
        self.setWindowIcon(QIcon('assets/app_icon.png'))
        self.show()

        self.threadpool = QtCore.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


    def createPieChart(self):   
        self.series = QPieSeries()
        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chartView = QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.Antialiasing)
        #self.setMinimumHeight(800)

    @staticmethod
    def single_split(audio_wav, from_sec, to_sec):
        return audio_wav[from_sec:to_sec]
    
    @staticmethod
    def multiple_split(audio_wav, sr, sec_per_split):
        chunks = []
        total_dur_sec = int(len(audio_wav) / sr)
        for i in range(0, total_dur_sec, sec_per_split):
            start, end = i*sr, (i+sec_per_split)*sr
            chunks.append(audio_wav[start:end])
        return chunks
    
    @staticmethod
    def stereo_to_mono(audiodata):
        d = audiodata.sum(axis=0) / 2
        return torch.tensor(d, dtype=torch.float64)
    
    @staticmethod
    def resample_audio(audiodata, sample_rate, resample_rate=16000):
        resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=audiodata.dtype)
        resampled_waveform = resampler(audiodata)
        return resampled_waveform
    
    @staticmethod
    def convertToAudio(video_path):  
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(video_path.replace(".mp4", ".wav"))
        video.close()
        audio.close()
    
    @staticmethod
    def prepare_data(wavform, feature_extractor=feature_extractor):
        input = feature_extractor(
            wavform, normalize=True, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        return input.input_values[0].to(device)

    def selectFile(self):
        self.selectedFileText.setText(self.info)
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                        'Open File',
                                                        './',
                                                        'Audio Files (*.wav *.mp3 *.m4a);;Video Files (*.mp4)')
        if not file:
            return
        else:
            self.dumpButton.setDisabled(True)
            print(file)
            self.series.clear()
            self.chart.setTitle("")
            '''
            if not file.endswith("wav") and not file.endswith("mp3") and not file.endswith("m4a") and not file.endswith("mp4") and not file.endswith("webm"):
                msgBox = QtWidgets.QMessageBox()
                msgBox.setText("The doc")
                msgBox.exec();
            '''
            if file.endswith(".mp4"):
               self.mediaPlayer.setMedia(
                QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))
               self.convertToAudio(video_path=file)
               file = file.replace(".mp4", ".wav")
            else:
                self.mediaPlayer.setMedia(
                QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))
        self.selectedFileText.setText("Выбранный файл: " + file)
        self.fileName = file


    def mediastateChanged(self, state):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.slider.setValue(position)
 
 
    def durationChanged(self, duration):
        self.slider.setRange(0, duration)

    def setSlidePosition(self, position):
        self.mediaPlayer.setPosition(position)

    def play(self):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def progress_fn(self, n):
        self.pbar.setValue(n)
        print("%d%% done" % n)
    
    def threadFinished(self):
        self.selectFileButton.setEnabled(True)
        self.classifyButton.setEnabled(True)
        self.progressBar.hide()
        print("Thread finished!")


    def output_to_json(self, emo_res):
        res = {"analysis_result":
               {"file_id":"21412",
                "file_path": self.fileName,
                "emotions":{"positive": round(self.series.slices()[0].percentage(),2),
                            "angry": round(self.series.slices()[1].percentage(),2),
                            "neutral": round(self.series.slices()[2].percentage(),2),
                            "sad": round(self.series.slices()[3].percentage(),2),
                            "other": round(self.series.slices()[4].percentage(),2)
                            }
                }
              }
        
        with open(f"./{os.path.basename(self.fileName).replace('.wav','.json')}", "w") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Отчет был успешно сохранен в текущей директории")
        msgBox.exec()

    def classifyEmotion(self, progress_callback):
        emo_res = {'neutral': 0,'angry': 0, 'positive': 0, 'sad': 0, 'other': 0}
        print(self.fileName)
        waveform, sample_rate = torchaudio.load(self.fileName, normalize=True)
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        waveform = self.stereo_to_mono(waveform)
        print(len(waveform))
        print(waveform)
        if len(waveform) / 16000 > 15:
           chunk_length = 10 * 16000
           chunks = [waveform[i:i+chunk_length] for i in range(0, len(waveform), chunk_length)]
           inputs = list(map(self.prepare_data,chunks))
        else:
            inputs = waveform.unsqueeze(dim=0).map(self.prepare_data,
                                    fn_kwargs={"feature_extractor": feature_extractor})
                                                            
        for i, input in enumerate(inputs):
            logits = model(input.unsqueeze(dim=0)).logits
            predictions = torch.argmax(logits, dim=-1)
            predicted_emotion = num2emotion[predictions.cpu().numpy()[0]]
            emo_res[predicted_emotion] += 1
            print(predicted_emotion)
            progress_callback.emit((i+1)*int(100/len(inputs)))

        return emo_res

    def print_output(self, emo_res):
        self.pbar.hide()

        self.emo_res = emo_res
        self.dumpButton.setEnabled(True)

        if self.firstDetect:
            self.series.append("Радость", 0)
            self.series.append("Злость", 0)
            self.series.append("Спокойствие", 0)
            self.series.append("Грусть", 0)
            self.series.append("Другое", 0)
            self.chart.addSeries(self.series)
            self.chart.setAnimationOptions(QChart.SeriesAnimations)
            self.chart.setTitle("Соотношение эмоций")
            
        self.series.setLabelsVisible(True)
        self.series.slices()[0].setValue(emo_res["positive"])
        self.series.slices()[1].setValue(emo_res["angry"])
        self.series.slices()[2].setValue(emo_res["neutral"])
        self.series.slices()[3].setValue(emo_res["sad"])
        self.series.slices()[4].setValue(emo_res["other"])

        self.series.slices()[0].setLabel("Радость: %.1f%%"  %(self.series.slices()[0].percentage()*100))
        self.series.slices()[1].setLabel("Злость: %.1f%%"  %(self.series.slices()[1].percentage()*100))
        self.series.slices()[2].setLabel("Спокойствие: %.1f%%"  %(self.series.slices()[2].percentage()*100))
        self.series.slices()[3].setLabel("Грусть: %.1f%%"  %(self.series.slices()[3].percentage()*100))
        self.series.slices()[4].setLabel("Другое: %.1f%%"  %(self.series.slices()[4].percentage()*100))

        for i in range(len(self.series.slices())):
            print(self.series.slices()[i].percentage())
            if self.series.slices()[i].percentage() < 0.009:
                self.series.slices()[i].setLabelVisible(False)


    def thread_complete(self):
        print("THREAD COMPLETE!")

    def detect(self):
        self.pbar.show()
        worker = Worker(self.classifyEmotion,"!!!DETECT!!!")
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        
        self.threadpool.start(worker)

    def dumpToFile(self):
        self.output_to_json(self.emo_res)
    
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    mainWindew = MainWindow()
    sys.exit(app.exec_())
