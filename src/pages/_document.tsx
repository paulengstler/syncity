import NextDocument, { Html, Head, Main, NextScript } from 'next/document'
import { ColorModeScript } from '@chakra-ui/react'

import { title, description } from 'data'

import theme from 'theme'

export default class Document extends NextDocument {
  render() {
    return (
      <Html>
        <Head />
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta property="og:title" content={title} key="ogtitle" />
        <meta property="og:description" content={description} key="ogdesc" />
        <meta property="og:site_name" content={title} key="ogsitename" />
        <meta property="og:image" content="https://research.paulengstler.com/syncity/images/og_image.jpg" />
        <meta property="og:url" content="https://research.paulengstler.com/syncity/" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta charSet="utf-8" />
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    )
  }
}
