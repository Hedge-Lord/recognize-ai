import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-wasm';
import * as faceapi from '@vladmandic/face-api';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true
})
export class AppComponent implements AfterViewInit, OnDestroy {
  title = 'webcam-facial-recognition';
  @ViewChild('webcamVideo') webcamVideo!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;
  private stream: MediaStream | null = null;

  async ngAfterViewInit(): Promise<void> {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/assets/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/assets/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/assets/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/assets/models'), 
    faceapi.nets.faceExpressionNet.loadFromUri('/assets/models') 
  ]);

    try {
      await tf.setBackend('webgl');
    } catch(error) {
      console.warn('WebGL not available, fallback to WASM');
      await tf.setBackend('wasm');
    }

    await tf.ready();
    // console.log("done");
  }

  ngOnDestroy(): void {
    this.stopWebcam();
  }

  startWebcam(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          this.stream = stream; 
          if (this.webcamVideo.nativeElement) {
            this.webcamVideo.nativeElement.srcObject = stream;
            this.webcamVideo.nativeElement.play().catch(console.error); 
          }
        })
        .catch(console.error);
    }
  }

  stopWebcam(): void {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop()); 
      if (this.webcamVideo.nativeElement) {
        this.webcamVideo.nativeElement.srcObject = null; 
      }
      this.stream = null; 
    }
  }
captureImage(): void {
  if (this.webcamVideo && this.canvas) {
    const context = this.canvas.nativeElement.getContext('2d');
    if (context) {
      const videoElement = this.webcamVideo.nativeElement;
      this.canvas.nativeElement.width = videoElement.videoWidth;
      this.canvas.nativeElement.height = videoElement.videoHeight;

      context.fillStyle = 'rgba(0, 0, 0, 0.5)'; 
      context.fillRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
      context.font = '20px Arial';
      context.fillStyle = 'white';
      context.fillText('Loading...', this.canvas.nativeElement.width / 2, this.canvas.nativeElement.height / 2);
      context.drawImage(videoElement, 0, 0);


      this.detectFaces().then(() => {
      }).catch(error => {
        console.error("Error during face detection:", error);
        context.fillStyle = 'rgba(0, 0, 0, 0.7)'; 
        context.fillRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
        context.fillStyle = 'red';
        context.fillText('Error processing the image', this.canvas.nativeElement.width / 2, this.canvas.nativeElement.height / 2);
      });
    }
  }
}

async detectFaces() {
  const detectionsWithAgeAndGender = await faceapi.detectAllFaces(
    this.canvas.nativeElement,
    new faceapi.TinyFaceDetectorOptions()
  ).withFaceLandmarks().withFaceExpressions().withAgeAndGender();

  const context = this.canvas.nativeElement.getContext('2d');
  if (context && detectionsWithAgeAndGender) {

    detectionsWithAgeAndGender.forEach(detection => {
      const { age, gender, genderProbability } = detection;
      
      const box = detection.detection.box;
      const text = [
        `${faceapi.utils.round(age, 0)} years`,
        `${gender} (${faceapi.utils.round(genderProbability)})`
      ];
      const drawOptions = {
        label: text.join(' '),
        boxColor: 'blue',
        lineWidth: 2
      };
      const drawBox = new faceapi.draw.DrawBox(box, drawOptions);
      drawBox.draw(this.canvas.nativeElement);
    
      faceapi.draw.drawFaceLandmarks(this.canvas.nativeElement, detection.landmarks);
    
      const minConfidence = 0.5;
      faceapi.draw.drawFaceExpressions(this.canvas.nativeElement, detectionsWithAgeAndGender, minConfidence);
    });
  }
}

@ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

onFileSelected(): void {
  const file = this.fileInput.nativeElement.files?.[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = async (e: ProgressEvent<FileReader>) => {
    const img = new Image();
    img.src = e.target!.result as string;
    img.onload = async () => {
      this.canvas.nativeElement.width = img.width;
      this.canvas.nativeElement.height = img.height;
      const context = this.canvas.nativeElement.getContext('2d');
      context!.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
      context!.drawImage(img, 0, 0, img.width, img.height);

      await this.detectFaces(); // You might need to adjust this for static images
    };
  };
  reader.readAsDataURL(file);
}
}
