from tts_generator import YTSpeechDataGenerator
generator = YTSpeechDataGenerator(dataset_name='indotts', lang='id')
generator.prepare_dataset('links.txt')