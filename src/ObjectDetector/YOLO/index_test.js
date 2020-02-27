// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT


const YOLO_DEFAULTS = {
  filterBoxesThreshold: 0.01,
	IOUThreshold: 0.4,
	classProbThreshold: 0.4,
	modelSize: [416, 416],
	maxOutput: 10,
};

describe('YOLO', () => {
  let yolo;

  async function getRobin() {
    const img = new Image();
    img.crossOrigin = '';
    img.src = 'https://cdn.jsdelivr.net/gh/ml5js/ml5-library@development/assets/bird.jpg';
    await new Promise((resolve) => { img.onload = resolve; });
    return img;
  }

  async function getImageData() {
    const arr = new Uint8ClampedArray(40000);

    // Iterate through every pixel
    for (let i = 0; i < arr.length; i += 4) {
      arr[i + 0] = 0;    // R value
      arr[i + 1] = 190;  // G value
      arr[i + 2] = 0;    // B value
      arr[i + 3] = 255;  // A value
    }

    // Initialize a new ImageData object
    const img = new ImageData(arr, 200);
    return img;
  }

  describe('object detector with yolo', () =>{
    beforeAll(async () => {
      jasmine.DEFAULT_TIMEOUT_INTERVAL = 100000;
      yolo = await ml5.objectDetector('yolo', { disableDeprecationNotice: true, ...YOLO_DEFAULTS });
    });
  
    it('instantiates the YOLO classifier with defaults', () => {
      expect(yolo.IOUThreshold).toBe(YOLO_DEFAULTS.IOUThreshold);
      expect(yolo.classProbThreshold).toBe(YOLO_DEFAULTS.classProbThreshold);
      expect(yolo.filterBoxesThreshold).toBe(YOLO_DEFAULTS.filterBoxesThreshold);
    });
  
    it('detects a robin', async () => {

      const robin = await getRobin();
      const detection = await yolo.detect(robin);

      expect(detection[0].label).toBe('bird');
    });
  
    it('detects takes ImageData', async () => {
      const img = await getImageData();
      const detection = await yolo.detect(img);
      expect(detection).toEqual([]);
    });
  })
  
});
