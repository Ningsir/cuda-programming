#ifndef BOOK_CODES_UTILS_H_
#define BOOK_CODES_UTILS_H_

#include <sys/time.h>

double getCurrentTime()
{
	timeval t;
	gettimeofday(&t, nullptr);
	return static_cast<double>(t.tv_sec) * 1000 +
		   static_cast<double>(t.tv_usec) / 1000;
}

#endif // BOOK_CODES_UTILS_H_
