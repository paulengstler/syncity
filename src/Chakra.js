// e.g. src/Chakra.js
// a) import `ChakraProvider` component as well as the storageManagers
import {
    ChakraProvider,
    cookieStorageManagerSSR,
    localStorageManager,
  } from '@chakra-ui/react'

import '@fontsource/pt-serif/400.css'
import '@fontsource/pt-serif/700.css'
//import '@fontsource/lexend-exa/400.css'
//import '@fontsource/lexend-exa/700.css'
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/600.css'
import '@fontsource/ibm-plex-sans/700.css'

import theme from 'theme'

export function Chakra({ cookies, children }) {
  // b) Pass `colorModeManager` prop
  const colorModeManager =
    typeof cookies === 'string'
      ? cookieStorageManagerSSR(cookies)
      : localStorageManager

  return (
    <ChakraProvider colorModeManager={colorModeManager} theme={theme}>
      {children}
    </ChakraProvider>
  )
}

// also export a reusable function getServerSideProps
export function getServerSideProps({ req }) {
  return {
    props: {
      // first time users will not have any cookies and you may not return
      // undefined here, hence ?? is necessary
      cookies: req.headers.cookie ?? '',
    },
  }
}
