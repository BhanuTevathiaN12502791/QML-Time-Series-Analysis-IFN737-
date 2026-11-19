import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from fetch_event_data import extract_archive


class ArchiveExtractionTests(unittest.TestCase):
    def archive(self, directory, **changes):
        row = dict(NB_SCATS_SITE=100, NB_DETECTOR=2, QT_INTERVAL_COUNT='2025-12-25',
                   CT_RECORDS=96, CT_ALARM_24HOUR=0, QT_VOLUME_24HOUR=0)
        row.update({f'V{i:02d}': 0 for i in range(96)})
        row.update(changes)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
        # A different detector must not be included even though its site matches.
        writer.writerow({**row, 'NB_DETECTOR': 99})
        path = Path(directory)/'sample.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('README.txt', 'skip non-CSV members')
            archive.writestr('day.csv', '\ufeff'+stream.getvalue())
        return path

    def test_zero_counts_and_fixed_aest_interval_end_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.archive(directory)
            daily, points = extract_archive(path, ['100:2'])
        self.assertEqual(len(daily), 1)
        self.assertEqual(len(points), 96)
        self.assertEqual(points[0]['volume'], 0)
        self.assertEqual(str(points[0]['timestamp']), '2025-12-25 00:15:00+10:00')
        self.assertEqual(str(points[-1]['timestamp']), '2025-12-26 00:00:00+10:00')

    def test_alarm_or_incomplete_source_day_masks_every_interval(self):
        for changes in [dict(CT_RECORDS=95), dict(CT_ALARM_24HOUR=1)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                daily, points = extract_archive(self.archive(directory, **changes), ['100:2'])
                self.assertFalse(daily[0]['usable'])
                self.assertTrue(all(point['volume'] is None for point in points))

    def test_invalid_and_negative_values_mask_only_their_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            _, points = extract_archive(self.archive(directory, V00=-1, V01='missing', V02=5), ['100:2'])
        self.assertIsNone(points[0]['volume'])
        self.assertIsNone(points[1]['volume'])
        self.assertEqual(points[2]['volume'], 5.)


if __name__ == '__main__':
    unittest.main()
